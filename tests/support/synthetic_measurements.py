"""Small authored contract cases. These are not model outputs or benchmark evidence."""

from copy import deepcopy

REF = "product_1:artifact:final_response"


def claim(index, *, refs=None, judgment="supported"):
    return {
        "claim": f"Synthetic statement {index}",
        "claim_refs": [REF],
        "claim_location": f"Paragraph {index}",
        "evidence_refs": list(refs if refs is not None else [REF]),
        "observation": f"Synthetic observation {index}",
        "judgment": judgment,
        "verdict_relevance": "material",
        "relevance_reason": "Required by the synthetic task",
    }


def assessment(claims, *, status="fail", refs=None):
    return {
        "claims": claims,
        "status": status,
        "reason": "Synthetic assessment of the supplied text",
        "gap": "Explain the missing detail" if status != "pass" else "",
        "evidence_refs": list(refs if refs is not None else [REF]),
    }


def judgment_replies():
    previews = ["product_1:source:2", "product_1:source:3"]
    first = assessment([claim(i, refs=previews if i == 1 else None) for i in range(5)])
    expanded = assessment([claim(0), claim(1, refs=previews)], refs=previews)
    del expanded["claims"][1]["judgment"]
    stale_repair = deepcopy(first)
    return first, expanded, stale_repair


def wrapped_reply():
    preview = ["product_1:source:2"]
    claims = [
        claim(
            i,
            refs=preview if i == 1 else None,
            judgment="unverified" if i == 1 else "supported",
        )
        for i in range(5)
    ]
    return {
        "response": assessment(claims, status="unknown", refs=preview),
        "inquiry_result": {
            "status": "unknown",
            "reason": "Read the full document",
            "gap": "Need full text",
            "evidence_refs": preview,
        },
    }


def scalar_inquiries():
    return [
        {
            "name": status,
            "test_id": "synthetic_inquiry",
            "evidence_refs": [REF],
            "response": {
                **assessment([claim(0, judgment=judgment)], status=status),
                "inquiry_result": status,
            },
        }
        for status, judgment in [("pass", "supported"), ("unknown", "unverified")]
    ]
