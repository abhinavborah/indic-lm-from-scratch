"""Surface-form templates for the synthetic reasoning dataset.

Four premise phrasings and four question phrasings, all comparative-only
(no numbers ever appear in the rendered text; only relative order matters).
Index 0-2 of each list are the trained forms, index 3 is held out entirely
for the "unseen wording" test split, per Lec09's split ratio (write four
surface forms, train on three, hold the fourth back).

All comparatives use gender-invariant words (ज़्यादा, कम, आगे, बेहतर) rather
than gendered adjectives (लंबा/लंबी), since the person-name pool mixes
genders and templates must not depend on knowing a name's gender.
"""

DOMAINS = {
    "height": {"attr_noun": "ऊंचाई", "entity_pool": "person"},
    "age": {"attr_noun": "उम्र", "entity_pool": "person"},
    "price": {"attr_noun": "कीमत", "entity_pool": "object"},
    "quantity": {"attr_noun": "मात्रा", "entity_pool": "object"},
}

PREMISE_FORMS = [
    "{bigger} की {attr} {smaller} से ज़्यादा है।",
    "{smaller} की {attr} {bigger} से कम है।",
    "अगर हम {bigger} और {smaller} की {attr} देखें, तो {bigger} आगे है।",
    "{attr} के मामले में, {bigger} की स्थिति {smaller} से बेहतर है।",
]

QUESTION_FORMS = [
    "इनमें से {superlative} कौन है, {names}?",
    "{names} में से {superlative} कौन सा है?",
    "बताइए, {names} में से {superlative} कौन है।",
    "{superlative} विकल्प कौन सा है, यह बताएं: {names}।",
]

TRAIN_FORM_IDS = (0, 1, 2)
HELD_FORM_ID = 3


def superlative_phrase(attr_noun, direction, num_entities):
    """direction: 'largest' or 'smallest'. Drops the 'सबसे' (-most) prefix for
    a pairwise (2-entity) comparison, since 'सबसे' on exactly two items reads
    as informal/awkward; keeps it for three-or-more-entity chains."""
    word = "ज़्यादा" if direction == "largest" else "कम"
    prefix = "सबसे " if num_entities > 2 else ""
    return f"{prefix}{word} {attr_noun} वाला"


def join_names(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " और " + names[-1]


def render_premise(form_id, bigger, smaller, attr_noun):
    return PREMISE_FORMS[form_id].format(bigger=bigger, smaller=smaller, attr=attr_noun)


def render_question(form_id, names, attr_noun, direction):
    phrase = superlative_phrase(attr_noun, direction, len(names))
    return QUESTION_FORMS[form_id].format(superlative=phrase, names=join_names(names))
