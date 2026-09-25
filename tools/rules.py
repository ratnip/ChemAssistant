def check_lipinski(descriptors: dict) -> dict:
    violations = []

    if descriptors["mw"] > 500:
        violations.append("MW > 500")

    if descriptors["logp"] > 5:
        violations.append("LogP > 5")

    if descriptors["hbd"] > 5:
        violations.append("HBD > 5")

    if descriptors["hba"] > 10:
        violations.append("HBA > 10")

    return {
        "lipinski_violations": len(violations),
        "violations": violations
    }