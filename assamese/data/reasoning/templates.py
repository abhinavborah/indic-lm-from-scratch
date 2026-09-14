"""Surface-form templates for the synthetic reasoning dataset.

Four premise phrasings and four question phrasings, all comparative-only
(no numbers ever appear in the rendered text; only relative order matters).
Index 0-2 of each list are the trained forms, index 3 is held out entirely
for the "unseen wording" test split, per Lec09's split ratio (write four
surface forms, train on three, hold the fourth back).

Uses Assamese-specific grammar (তকৈ as the comparative postposition, ৰ as
the genitive marker, থকা for "the one having") rather than a Bengali-style
comparative (যেমন চেয়ে/থেকে), since Assamese and Bengali share the Unicode
block but are different languages. Independent from
hindi/data/reasoning/templates.py, not a transliteration of it.

Both ৰ (genitive) and তকৈ (comparative "than") are bound suffixes in
Assamese orthography and attach directly to the preceding word with no
space (নীলিমাৰ, গৌৰৱতকৈ), unlike Hindi's की/से which are separate words.
Verified against the real Phase 1 Assamese corpus and the trained
tokenizer 2026-09-13: the attached form tokenizes cleanly
(['_নীল','িম','াৰ']), the spaced form splits ৰ off as its own token and is
not the corpus's actual usage (the corpus's rare standalone " ৰ "/" তকৈ "
hits are math/logic notation, e.g. "1 তকৈ ডাঙৰ", not genitive/comparative
use). থকা's spaced form (দাম থকা) is the one that tokenizes cleanly and
is correct as written; it is a free word, not a suffix, so it keeps its
space.
"""

DOMAINS = {
    "height": {"attr_noun": "উচ্চতা", "entity_pool": "person"},
    "age": {"attr_noun": "বয়স", "entity_pool": "person"},
    "price": {"attr_noun": "দাম", "entity_pool": "object"},
    "quantity": {"attr_noun": "পৰিমাণ", "entity_pool": "object"},
}

PREMISE_FORMS = [
    "{bigger}ৰ {attr} {smaller}তকৈ বেছি।",
    "{smaller}ৰ {attr} {bigger}তকৈ কম।",
    "যদি আমি {bigger} আৰু {smaller}ৰ {attr} চাওঁ, তেন্তে {bigger} আগত আছে।",
    "{attr}ৰ ক্ষেত্ৰত, {bigger}ৰ অৱস্থা {smaller}তকৈ ভাল।",
]

# Quantitative/direct-equality register only (সমান) -- not সমতুল্য (semantic
# equivalence), সমানতা (social/legal equality), or সমতা (abstract fairness);
# those describe different task types entirely (paraphrase judgment, civics
# content, philosophical prose), none of which involve a comparison chain.
EQUAL_WORD = "সমান"
EQUAL_PREMISE_FORM = "{a} আৰু {b}ৰ {attr} সমান।"


def render_equal_premise(a, b, attr_noun):
    return EQUAL_PREMISE_FORM.format(a=a, b=b, attr=attr_noun)

QUESTION_FORMS = [
    "এইবোৰৰ ভিতৰত {superlative} জন কোন, {names}?",
    "{names}ৰ ভিতৰত {superlative} কোনটো?",
    "কওক, {names}ৰ ভিতৰত {superlative} কোন।",
    "{superlative} বিকল্পটো কোনটো, ইয়াক কওক: {names}।",
]

TRAIN_FORM_IDS = (0, 1, 2)
HELD_FORM_ID = 3


def superlative_phrase(attr_noun, direction, num_entities):
    """direction: 'largest' or 'smallest'. Drops the superlative prefix
    (সৰ্বাধিক/সৰ্বনিম্ন) for a pairwise (2-entity) comparison in favor of the
    plain comparative (বেছি/কম), since a strict superlative on exactly two
    items reads as unnatural; keeps it for three-or-more-entity chains."""
    if num_entities > 2:
        word = "সৰ্বাধিক" if direction == "largest" else "সৰ্বনিম্ন"
    else:
        word = "বেছি" if direction == "largest" else "কম"
    return f"{word} {attr_noun} থকা"


def join_names(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " আৰু " + names[-1]


def render_premise(form_id, bigger, smaller, attr_noun):
    return PREMISE_FORMS[form_id].format(bigger=bigger, smaller=smaller, attr=attr_noun)


def render_question(form_id, names, attr_noun, direction):
    phrase = superlative_phrase(attr_noun, direction, len(names))
    return QUESTION_FORMS[form_id].format(superlative=phrase, names=join_names(names))
