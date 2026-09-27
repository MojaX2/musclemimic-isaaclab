"""Full-body failure criterion beyond the rigid G1 trunk's root orientation."""


def torso_collapsed(head_gap, initial_head_gap):
    """Allow moderate lean, but reject a severely folded or inverted trunk."""
    return head_gap < .6 * initial_head_gap
