def find_predicate_for_threshold(threshold):
    predicate_map = {
        int: scalar_predicate,
        float: scalar_predicate,
        str: categorical_predicate,
        tuple: contains_predicate
    }
    predicate = predicate_map.get(type(threshold))
    if predicate is None:
        raise ValueError(f"Unknown threshold type: {type(threshold)}")
    return predicate


def scalar_predicate(feature, threshold):
    return feature >= threshold


def categorical_predicate(feature, threshold):
    return feature == threshold


def contains_predicate(features, threshold):
    return threshold in set(features)

