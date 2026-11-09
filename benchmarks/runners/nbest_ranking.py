"""Order persisted candidates using the decoder's actual sequence ranker."""


def rank_candidates(candidates, ranker):
    remaining = list(candidates)
    ordered = []
    while remaining:
        selected = ranker.rank([[c['tokens'] for c in remaining]],
                               [[c['sum_logprob'] for c in remaining]])[0]
        ordered.append(remaining.pop(selected))
    return ordered
