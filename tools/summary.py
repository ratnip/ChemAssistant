def summarize_validation(validation_results: list[dict]) -> dict:
    total = len(validation_results)

    valid = sum(
        1 for result in validation_results
        if result["valid"]
    )

    invalid = total - valid

    return {
        "total_molecules": total,
        "valid_molecules": valid,
        "invalid_molecules": invalid
    }


def summarize_descriptors(descriptor_results: list[dict]) -> dict:
    valid_results = [
        r for r in descriptor_results
        if "error" not in r
    ]

    if not valid_results:
        return {
            "valid_molecules": 0,
            "descriptor_statistics": {}
        }

    descriptor_names = [
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "heavy_atoms",
        "formal_charge",
    ]

    statistics = {}

    for name in descriptor_names:
        values = [r[name] for r in valid_results]

        statistics[name] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "valid_molecules": len(valid_results),
        "descriptor_statistics": statistics,
    }