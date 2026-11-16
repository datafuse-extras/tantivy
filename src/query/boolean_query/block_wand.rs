use crate::query::term_query::TermScorer;
use crate::query::Scorer;
use crate::{DocId, DocSet, Score, TERMINATED};

/// Specialized version of [`block_max_maxscore`] for a single scorer.
/// In this case, the algorithm is simple, readable and faster (~ x3)
/// than the generic algorithm.
/// The algorithm behaves as follows:
/// - While we don't hit the end of the docset:
///   - While the block max score is under the `threshold`, go to the next block.
///   - On a block, advance until the end and execute `callback` when the doc score is greater or
///     equal to the `threshold`.
pub fn block_wand_single_scorer(
    mut scorer: TermScorer,
    mut threshold: Score,
    callback: &mut dyn FnMut(u32, Score) -> Score,
) {
    let mut doc = scorer.doc();
    loop {
        // We position the scorer on a block that can reach
        // the threshold.
        while scorer.block_max_score() < threshold {
            let last_doc_in_block = scorer.last_doc_in_block();
            if last_doc_in_block == TERMINATED {
                return;
            }
            doc = last_doc_in_block + 1;
            scorer.seek_block(doc);
        }
        // Seek will effectively load that block.
        doc = scorer.seek(doc);
        if doc == TERMINATED {
            break;
        }
        loop {
            let score = scorer.score();
            if score > threshold {
                threshold = callback(doc, score);
            }
            debug_assert!(doc <= scorer.last_doc_in_block());
            if doc == scorer.last_doc_in_block() {
                break;
            }
            doc = scorer.advance();
            if doc == TERMINATED {
                return;
            }
        }
        doc += 1;
        scorer.seek_block(doc);
    }
}

/// Block-max MaxScore (Turtle & Flood's MaxScore with per-window block-max bounds, as in Lucene's
/// `MaxScoreBulkScorer`) for a top-k disjunction of term scorers.
///
/// Documents are visited in windows that end at the first block boundary of a scorer that can
/// still produce a competitive document on its own (a globally essential scorer). In a window,
/// every scorer is bounded by the block max of its block covering the window, or by its maximum
/// score if no single block covers it. The scorers are ordered by that bound; the longest prefix
/// whose bounds sum to at most the threshold is non-essential for the window: a document that only
/// contains those terms cannot enter the top-k, so only the other scorers produce candidates, in
/// doc order. A window without essential scorers is skipped without decoding it.
///
/// For a candidate, the exact essential score is summed, then the non-essential scorers are
/// probed from the highest bound down, and the candidate is dropped as soon as its score plus the
/// bounds of what is left cannot exceed the threshold.
///
/// Unlike block WAND, frequent, low-idf terms never drive the iteration once the threshold has
/// risen above their bound, and no per-candidate sort of the scorers is needed.
pub fn block_max_maxscore(
    mut scorers: Vec<TermScorer>,
    mut threshold: Score,
    callback: &mut dyn FnMut(u32, Score) -> Score,
) {
    scorers.retain(|scorer| scorer.doc() != TERMINATED);
    if scorers.is_empty() {
        return;
    }
    scorers.sort_by(|left, right| {
        left.max_score()
            .partial_cmp(&right.max_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let num_scorers = scorers.len();
    let max_scores: Vec<Score> = scorers.iter().map(|scorer| scorer.max_score()).collect();
    // global_bounds[i]: sum of the maximum scores of scorers[..=i].
    let mut global_bounds = Vec::with_capacity(num_scorers);
    let mut sum: Score = 0.0;
    for max_score in &max_scores {
        sum += max_score;
        global_bounds.push(sum);
    }
    // Per window: scorer ordinals ordered by their window bound, and the prefix sums of the bounds.
    let mut order: Vec<usize> = (0..num_scorers).collect();
    let mut bounds: Vec<Score> = vec![0.0; num_scorers];
    let mut prefix: Vec<Score> = vec![0.0; num_scorers];

    let mut window_min: DocId = 0;
    loop {
        // scorers[..num_global_non_essential] cannot produce a competitive document on their own,
        // even with their maximum scores.
        let mut num_global_non_essential = 0;
        while num_global_non_essential < num_scorers
            && global_bounds[num_global_non_essential] <= threshold
        {
            num_global_non_essential += 1;
        }
        if num_global_non_essential == num_scorers {
            return;
        }
        // The window ends at the first block end of a globally essential scorer.
        let mut window_max = TERMINATED;
        let mut next_doc = TERMINATED;
        for scorer in &mut scorers[num_global_non_essential..] {
            if scorer.doc() == TERMINATED {
                continue;
            }
            if scorer.doc() < window_min {
                scorer.seek_block(window_min);
            }
            window_max = window_max.min(scorer.last_doc_in_block());
            next_doc = next_doc.min(scorer.doc().max(window_min));
        }
        if next_doc == TERMINATED {
            return;
        }
        window_min = window_min.max(next_doc.min(window_max));
        for ord in 0..num_scorers {
            let scorer = &mut scorers[ord];
            bounds[ord] = if scorer.doc() == TERMINATED {
                0.0
            } else {
                if scorer.doc() < window_min {
                    scorer.seek_block(window_min);
                }
                if scorer.last_doc_in_block() >= window_max {
                    scorer.block_max_score().min(max_scores[ord])
                } else {
                    max_scores[ord]
                }
            };
        }
        order.sort_unstable_by(|&left, &right| {
            bounds[left]
                .partial_cmp(&bounds[right])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut acc: Score = 0.0;
        for (i, &ord) in order.iter().enumerate() {
            acc += bounds[ord];
            prefix[i] = acc;
        }
        let mut num_non_essential = 0;
        while num_non_essential < num_scorers && prefix[num_non_essential] <= threshold {
            num_non_essential += 1;
        }
        if num_non_essential < num_scorers {
            for &ord in &order[num_non_essential..] {
                let scorer = &mut scorers[ord];
                if scorer.doc() < window_min {
                    scorer.seek(window_min);
                }
            }
            loop {
                let mut doc = TERMINATED;
                for &ord in &order[num_non_essential..] {
                    doc = doc.min(scorers[ord].doc());
                }
                if doc > window_max || doc == TERMINATED {
                    break;
                }
                let mut score: Score = 0.0;
                for &ord in &order[num_non_essential..] {
                    let scorer = &mut scorers[ord];
                    if scorer.doc() == doc {
                        score += scorer.score();
                    }
                }
                let mut competitive = true;
                for i in (0..num_non_essential).rev() {
                    if score + prefix[i] <= threshold {
                        competitive = false;
                        break;
                    }
                    let scorer = &mut scorers[order[i]];
                    if scorer.doc() < doc {
                        scorer.seek(doc);
                    }
                    if scorer.doc() == doc {
                        score += scorer.score();
                    }
                }
                if competitive && score > threshold {
                    threshold = callback(doc, score);
                    while num_non_essential < num_scorers && prefix[num_non_essential] <= threshold
                    {
                        num_non_essential += 1;
                    }
                }
                for &ord in &order[num_non_essential..] {
                    let scorer = &mut scorers[ord];
                    if scorer.doc() == doc {
                        scorer.advance();
                    }
                }
                if num_non_essential == num_scorers {
                    break;
                }
            }
        }
        if window_max == TERMINATED {
            return;
        }
        window_min = window_max + 1;
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    use proptest::prelude::*;

    use crate::query::score_combiner::SumCombiner;
    use crate::query::term_query::TermScorer;
    use crate::query::{Bm25Weight, BufferedUnionScorer, Scorer};
    use crate::{DocId, DocSet, Score, TERMINATED};

    struct Float(Score);

    impl Eq for Float {}

    impl PartialEq for Float {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }

    impl PartialOrd for Float {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Float {
        fn cmp(&self, other: &Self) -> Ordering {
            other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
        }
    }

    fn nearly_equals(left: Score, right: Score) -> bool {
        (left - right).abs() < 0.0001 * (left + right).abs()
    }

    fn compute_checkpoints_for_each_pruning(
        mut term_scorers: Vec<TermScorer>,
        n: usize,
    ) -> Vec<(DocId, Score)> {
        let mut heap: BinaryHeap<Float> = BinaryHeap::with_capacity(n);
        let mut checkpoints: Vec<(DocId, Score)> = Vec::new();
        let mut limit: Score = 0.0;

        let callback = &mut |doc, score| {
            heap.push(Float(score));
            if heap.len() > n {
                heap.pop().unwrap();
            }
            if heap.len() == n {
                limit = heap.peek().unwrap().0;
            }
            if !nearly_equals(score, limit) {
                checkpoints.push((doc, score));
            }
            limit
        };

        if term_scorers.len() == 1 {
            let scorer = term_scorers.pop().unwrap();
            super::block_wand_single_scorer(scorer, Score::MIN, callback);
        } else {
            super::block_max_maxscore(term_scorers, Score::MIN, callback);
        }
        checkpoints
    }

    fn compute_checkpoints_manual(
        term_scorers: Vec<TermScorer>,
        n: usize,
        max_doc: u32,
    ) -> Vec<(DocId, Score)> {
        let mut heap: BinaryHeap<Float> = BinaryHeap::with_capacity(n);
        let mut checkpoints: Vec<(DocId, Score)> = Vec::new();
        let mut scorer = BufferedUnionScorer::build(term_scorers, SumCombiner::default, max_doc);

        let mut limit = Score::MIN;
        loop {
            if scorer.doc() == TERMINATED {
                break;
            }
            let doc = scorer.doc();
            let score = scorer.score();
            if score > limit {
                heap.push(Float(score));
                if heap.len() > n {
                    heap.pop().unwrap();
                }
                if heap.len() == n {
                    limit = heap.peek().unwrap().0;
                }
                if !nearly_equals(score, limit) {
                    checkpoints.push((doc, score));
                }
            }
            scorer.advance();
        }
        checkpoints
    }

    const MAX_TERM_FREQ: u32 = 100u32;

    fn posting_list(max_doc: u32) -> BoxedStrategy<Vec<(DocId, u32)>> {
        (1..max_doc + 1)
            .prop_flat_map(move |doc_freq| {
                (
                    proptest::bits::bitset::sampled(doc_freq as usize, 0..max_doc as usize),
                    proptest::collection::vec(1u32..MAX_TERM_FREQ, doc_freq as usize),
                )
            })
            .prop_map(|(docset, term_freqs)| {
                docset
                    .iter()
                    .map(|doc| doc as u32)
                    .zip(term_freqs.iter().cloned())
                    .collect::<Vec<_>>()
            })
            .boxed()
    }

    #[expect(clippy::type_complexity)]
    fn gen_term_scorers(num_scorers: usize) -> BoxedStrategy<(Vec<Vec<(DocId, u32)>>, Vec<u32>)> {
        (1u32..100u32)
            .prop_flat_map(move |max_doc: u32| {
                (
                    proptest::collection::vec(posting_list(max_doc), num_scorers),
                    proptest::collection::vec(2u32..10u32 * MAX_TERM_FREQ, max_doc as usize),
                )
            })
            .boxed()
    }

    fn test_block_wand_aux(posting_lists: &[Vec<(DocId, u32)>], fieldnorms: &[u32]) {
        // We virtually repeat all docs 64 times in order to emulate blocks of 2 documents
        // and surface blogs more easily.
        const REPEAT: usize = 64;
        let fieldnorms_expanded = fieldnorms
            .iter()
            .cloned()
            .flat_map(|fieldnorm| std::iter::repeat_n(fieldnorm, REPEAT))
            .collect::<Vec<u32>>();

        let postings_lists_expanded: Vec<Vec<(DocId, u32)>> = posting_lists
            .iter()
            .map(|posting_list| {
                posting_list
                    .iter()
                    .cloned()
                    .flat_map(|(doc, term_freq)| {
                        (0_u32..REPEAT as u32).map(move |offset| {
                            (
                                doc * (REPEAT as u32) + offset,
                                if offset == 0 { term_freq } else { 1 },
                            )
                        })
                    })
                    .collect::<Vec<(DocId, u32)>>()
            })
            .collect::<Vec<_>>();

        let total_fieldnorms: u64 = fieldnorms_expanded
            .iter()
            .cloned()
            .map(|fieldnorm| fieldnorm as u64)
            .sum();
        let average_fieldnorm = (total_fieldnorms as Score) / (fieldnorms_expanded.len() as Score);
        let max_doc = fieldnorms_expanded.len();

        let term_scorers: Vec<TermScorer> = postings_lists_expanded
            .iter()
            .map(|postings| {
                let bm25_weight = Bm25Weight::for_one_term(
                    postings.len() as u64,
                    max_doc as u64,
                    average_fieldnorm,
                );
                TermScorer::create_for_test(postings, &fieldnorms_expanded[..], bm25_weight)
            })
            .collect();
        for top_k in 1..4 {
            let checkpoints_for_each_pruning =
                compute_checkpoints_for_each_pruning(term_scorers.clone(), top_k);
            let checkpoints_manual =
                compute_checkpoints_manual(term_scorers.clone(), top_k, max_doc as u32);
            assert_eq!(checkpoints_for_each_pruning.len(), checkpoints_manual.len());
            for (&(left_doc, left_score), &(right_doc, right_score)) in checkpoints_for_each_pruning
                .iter()
                .zip(checkpoints_manual.iter())
            {
                assert_eq!(left_doc, right_doc);
                assert!(nearly_equals(left_score, right_score));
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn test_block_wand_two_term_scorers((posting_lists, fieldnorms) in gen_term_scorers(2)) {
            test_block_wand_aux(&posting_lists[..], &fieldnorms[..]);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn test_block_wand_single_term_scorer((posting_lists, fieldnorms) in gen_term_scorers(1)) {
            test_block_wand_aux(&posting_lists[..], &fieldnorms[..]);
        }
    }

    #[test]
    fn test_fn_reproduce_proptest() {
        let postings_lists = &[
            vec![
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (6, 1),
                (7, 7),
                (8, 1),
                (10, 1),
                (12, 1),
                (13, 1),
                (14, 1),
                (15, 1),
                (16, 1),
                (19, 1),
                (20, 1),
                (21, 1),
                (22, 1),
                (24, 1),
                (25, 1),
                (26, 1),
                (28, 1),
                (30, 1),
                (31, 1),
                (33, 1),
                (34, 1),
                (35, 1),
                (36, 95),
                (37, 1),
                (39, 1),
                (41, 1),
                (44, 1),
                (46, 1),
            ],
            vec![
                (0, 5),
                (2, 1),
                (4, 1),
                (5, 84),
                (6, 47),
                (7, 26),
                (8, 50),
                (9, 34),
                (11, 73),
                (12, 11),
                (13, 51),
                (14, 45),
                (15, 18),
                (18, 60),
                (19, 80),
                (20, 63),
                (23, 79),
                (24, 69),
                (26, 35),
                (28, 82),
                (29, 19),
                (30, 2),
                (31, 7),
                (33, 40),
                (34, 1),
                (35, 33),
                (36, 27),
                (37, 24),
                (38, 65),
                (39, 32),
                (40, 85),
                (41, 1),
                (42, 69),
                (43, 11),
                (45, 45),
                (47, 97),
            ],
            vec![
                (2, 1),
                (4, 1),
                (7, 94),
                (8, 1),
                (9, 1),
                (10, 1),
                (12, 1),
                (15, 1),
                (22, 1),
                (23, 1),
                (26, 1),
                (27, 1),
                (32, 1),
                (33, 1),
                (34, 1),
                (36, 96),
                (39, 1),
                (41, 1),
            ],
        ];
        let fieldnorms = &[
            685, 239, 780, 564, 664, 827, 5, 56, 930, 887, 263, 665, 167, 127, 120, 919, 292, 92,
            489, 734, 814, 724, 700, 304, 128, 779, 311, 877, 774, 15, 866, 368, 894, 371, 982,
            502, 507, 669, 680, 76, 594, 626, 578, 331, 170, 639, 665, 186,
        ][..];
        test_block_wand_aux(postings_lists, fieldnorms);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn test_block_max_maxscore_three_term_scorers((posting_lists, fieldnorms) in gen_term_scorers(3)) {
            test_block_wand_aux(&posting_lists[..], &fieldnorms[..]);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]
        #[test]
        fn test_block_max_maxscore_five_term_scorers((posting_lists, fieldnorms) in gen_term_scorers(5)) {
            test_block_wand_aux(&posting_lists[..], &fieldnorms[..]);
        }
    }
}
