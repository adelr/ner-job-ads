"""
A set of utility functions for the NER pipeline

MIT License

Copyright (c) 2022 Adel Rahmani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import ast
from genericpath import exists
import time
import regex
import json
import pandas as pd
import seaborn as sns
import hashlib
import random

from IPython.display import HTML, display
from pathlib import Path
from functools import wraps
from collections import defaultdict
from itertools import chain
import copy

import spacy
from spacy.util import filter_spans
from spacy.tokens import Span
from spacy import displacy
from spacy.language import Language

from tqdm import tqdm_notebook

pattern_punctuation = regex.compile("[[:punct:]]")
pattern_multiple_blanks = regex.compile("\s+")
pattern_multiple_stars = regex.compile("\*+")

AVAILABLE_SPACY_MODELS = {m: f"en_core_web_{m}" for m in ["sm", "md", "lg", "trf"]}

_ENT_LABELS_REMOVE_DEFAULT = {
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "LAW",
    "MONEY",
    "NORP",
    "ORDINAL",
    # 'ORG',
    "PERCENT",
    "PERSON",
    # 'GPE',
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
}


def create_path(path, exist_ok=False):
    try:
        path.mkdir(parents=True, exist_ok=exist_ok)
        print(f"{path.as_posix()} created...")
    except FileExistsError:
        print(f"{path.as_posix()} already exists...")
        if not exist_ok:
            raise FileExistsError
    return


def pd2hash(df):
    return hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()


def timedelta2str(delta):
    c = delta.components
    days, hours, minutes, seconds = c.days, c.hours, c.minutes, c.seconds
    return f"{days}d{hours}h{minutes}m{seconds}s"


def compute_elapsed_time(func):
    """
    Decorator to compute the elapsed time and ouptut it as a formatted string
    """

    @wraps(func)
    def with_time(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = timedelta2str(pd.Timedelta(end - start, unit="sec"))
        return (result, elapsed)

    return with_time


def get_items_with_word(word, items=None):
    return [t for t in items if word in t]


def get_items_with_number_of_words(n_words=1, items=None):
    return [t for t in items if len(t.split()) == n_words]


def create_regex_from_sequence(seq, flags=0):
    expression = "|".join([f"({t})" for t in seq])
    pat = regex.compile(expression, flags=flags)
    return pat


def create_model_and_add_rules(
    model_type,
    components,
    remove_unused_entities=True,
    ENT_LABELS_REMOVE=None,
    disable="",
):

    assert model_type in AVAILABLE_SPACY_MODELS.keys()
    nlp = spacy.load(AVAILABLE_SPACY_MODELS[model_type], disable=disable)
    model = add_custom_ner_pipelines(
        nlp, components, remove_unused_entities, ENT_LABELS_REMOVE
    )
    return model


def add_custom_ner_pipelines(
    model, components, remove_unused_entities=True, ENT_LABELS_REMOVE=None
):
    for k in components:
        model.add_pipe(k, before="ner")
    if remove_unused_entities:
        if ENT_LABELS_REMOVE is None:
            # ENT_LABELS_REMOVE = _ENT_LABELS_REMOVE_DEFAULT
            ENT_LABELS_REMOVE = set()

        @Language.component("ner_removal")
        def ner_removal(doc):
            ents = list(doc.ents)
            for ent in ents:
                if ent.label_ in ENT_LABELS_REMOVE:
                    ents.remove(ent)
            ents = tuple(ents)
            doc.ents = ents
            return doc

        if len(ENT_LABELS_REMOVE) > 0:
            model.add_pipe("ner_removal", after="ner")
    return model


def create_spacy_lang_component(pattern=None, label=None):
    @Language.component(f"find_{label}")
    def find_entity(doc):
        # text = doc.text
        camp_ents = []
        original_ents = list(doc.ents)
        for match in pattern.finditer(doc.text, concurrent=True):
            start, end = match.span()
            span = doc.char_span(start, end)
            if span is not None:
                camp_ents.append((span.start, span.end, span.text))
        for ent in camp_ents:
            start, end, name = ent
            per_ent = Span(doc, start, end, label=label.upper())
            original_ents.append(per_ent)
        filtered = filter_spans(original_ents)
        doc.ents = filtered
        return doc

    return find_entity


def create_spacy_components_from_dict(pattern_dict=None):
    custom_spacy_components = {}
    for k, v in pattern_dict.items():
        custom_spacy_components[f"find_{k}"] = create_spacy_lang_component(
            pattern=v, label=k
        )
    return custom_spacy_components


def compare_models(
    ref_models=None,
    ref_labels=None,
    custom_model=None,
    data=None,
    max_length=None,
    k=5,
    sort_annot=False,
    random_state=0,
    options=None,
):

    if isinstance(ref_models, (list, tuple, set)):
        n_models = len(ref_models)
    elif isinstance(ref_models, spacy.lang.en.English):
        ref_models = tuple(
            [ref_models],
        )
        ref_labels = tuple(
            [ref_labels],
        )
    else:
        raise ValueError("Provide a reference spacy model.")

    model_types = [m.meta["name"].split("_")[-1] for m in ref_models]

    if max_length is not None:
        data_ = [_ for _ in data if len(_[0]) <= max_length]
        print(
            f"Filtered down to {len(data_)} document with length less than {max_length} characters..."
        )
    else:
        data_ = copy.deepcopy(data)

    if sort_annot:
        selection = sorted(data_, key=lambda x: len(x[1]["entities"]), reverse=True)[:k]
    else:
        random.seed(random_state)
        selection = random.choices(data_, k=k)

    for t, annos, doc_id in selection:
        # d_ref = ref_model(t)
        d_cust = custom_model(t)

        if len(d_cust.ents) > 0:

            for m, mlab, mtype in zip(ref_models, ref_labels, model_types):
                d_ref = m(t)
                display(HTML(f"<hr><h3>Reference ({mlab}):</h3>"))
                displacy.render(d_ref, style="ent", options=options)

            display(HTML(f"<hr><h3>Custom model:</h3>"))
            displacy.render(d_cust, style="ent", options=options)
            print("\n\n")
        # print(t, annos)


def save_ner_regex_to_json(pattern_dict=None, file=None):
    spacy_ner_components = defaultdict(dict)
    for k in pattern_dict:
        pat = pattern_dict.get(k)
        spacy_ner_components[k]["pattern"] = pat.pattern
        spacy_ner_components[k]["flags"] = pat.flags
    with open(file, "w") as f:
        json.dump(spacy_ner_components, f)
    return


def load_ner_regex_from_json(file=None):
    with open(file, "r") as f:
        component_dict = json.load(f)
    pat_dict = {}
    for k in component_dict:
        p, flags = component_dict[k]["pattern"], component_dict[k]["flags"]
        pat_dict[k] = regex.compile(p, flags=flags)
    return pat_dict


def get_model_ent_scores(model, sort_by_f=True):

    if "_" in model.meta["name"]:
        model_type = "_" + model.meta["name"].split("_")[-1]
    else:
        model_type = "_custom"

    _df = pd.DataFrame(model.meta["performance"]["ents_per_type"]).round(2).T
    if sort_by_f:
        _df = _df.sort_values(by=_df.columns[-1], ascending=False)
    else:
        _df = _df.sort_index()
    return _df


def save_spacy_ner_data_to_disk(data, path="./annotations_workbench", suffix=""):
    """
    Save annotated spaCy NER data to disk
    """
    df_ = pd.DataFrame(data, columns=["text", "entities", "doc_id"])
    filename = Path(path) / f"ner_annotation_data_{suffix}.csv.gz"
    df_.to_csv(filename, index=False)
    return


def load_spacy_data_from_csv(path):
    """
    Load an existing spaCy data file from disk
    """
    df_ = pd.read_csv(path)
    return [(r[0], ast.literal_eval(r[1]), r[2]) for _, r in df_.iterrows()]


def extract_ent_location_batch(texts, nlp=None, entities={"OCCUP", "EMPLOYER"}):
    docs = nlp.pipe(texts)
    return [
        (
            doc.text,
            {
                "entities": [
                    (e.start_char, e.end_char, e.label_)
                    for e in doc.ents
                    if e.label_ in entities
                ]
            },
        )
        for doc in docs
    ]


def extract_ent_location(text, nlp=None, entities={"OCCUP", "EMPLOYER"}):
    doc = nlp(text)
    ents = [
        (e.start_char, e.end_char, e.label_) for e in doc.ents if e.label_ in entities
    ]
    return (doc.text, {"entities": ents})


def entities_found(annotated_text, entities={"OCCUP", "EMPLOYER"}, method="ANY"):
    """
    Look for entities specified in the entities parameter.
    If method is ALL only returns samples containing all the entities,
    otherwise return samples with ANY of the entities.
    """
    d = annotated_text[1]["entities"]
    ents = {_[-1] for _ in d}
    if method == "ALL":
        return ents.issuperset(entities)
    else:
        return len(ents.intersection(entities)) > 0


@compute_elapsed_time
def build_annotations_from_sentences(
    data=None, nlp=None, entities={"OCCUP", "EMPLOYER"}, method="ANY"
):

    """
    Build the annotations by sentences from either the full document if sentence_data is None,
    or the precomputed sentences if the sentence level data is passed on.
    Keep track of the document id.
    """

    def annotate_sentences(sentences, **kwargs):
        annotations = [
            tuple(list(r) + [doc_id])
            for r in list(
                map(
                    lambda _: extract_ent_location(_, nlp=nlp, entities=entities),
                    sentences,
                )
            )
            if r[1]["entities"] and entities_found(r, entities=entities, method=method)
        ]
        return annotations

    sentence_data = True if "sentence" in data.columns else False

    annotated_data = []

    if not sentence_data:

        texts = data[["Id", "FullDescription"]].values

        for doc_id, t in tqdm_notebook(texts):

            sentences = [
                sent.text.strip()
                for sent in nlp(t).sents
                if pattern_punctuation.sub("", sent.text).strip()
            ]
            sentences = [s for s in sentences if s]

            annotations = annotate_sentences(
                sentences, nlp=nlp, entities=entities, method=method
            )

            if len(annotations) > 0:
                annotated_data.extend(annotations)
    else:

        doc_sents = data.groupby("doc_id")["sentence"].apply(list).reset_index().values

        for doc_id, sentences in tqdm_notebook(doc_sents):

            annotations = annotate_sentences(
                sentences, nlp=nlp, entities=entities, method=method
            )

            if len(annotations) > 0:
                annotated_data.extend(annotations)

    return annotated_data


@compute_elapsed_time
def build_annotations_from_docs(
    data=None, nlp=None, entities={"OCCUP", "EMPLOYER"}, method="ANY"
):

    annotated_data = []

    texts = data[["Id", "FullDescription"]].values

    for doc_id, t in tqdm_notebook(texts):

        annotations_ = extract_ent_location(t, nlp=nlp, entities=entities)

        annotations = (
            annotations_
            if entities_found(annotations_, entities=entities, method=method)
            else ""
        )

        if len(annotations) > 0:
            annotations = tuple(list(annotations) + [doc_id])
            annotated_data.append(annotations)
    return annotated_data


def get_annotation_metadata(annotated_data):
    df_ = pd.DataFrame(
        [{"doc_id": doc_id, "text": t} for t, _, doc_id in annotated_data]
    )
    df_["text_length"] = df_.text.str.len()
    entities = (
        pd.concat(
            [
                pd.value_counts([_[-1] for _ in d["entities"]])
                for _, d, _ in annotated_data
            ],
            axis=1,
        )
        .sort_index()
        .fillna(0)
        .astype(int)
        .T
    )
    entities = entities.assign(entity_count=entities.sum(axis=1))
    df_ = pd.concat((df_, entities), axis=1)
    df_ = df_.assign(entity_per_char=df_.entity_count / df_.text_length)
    return df_


def get_available_trained_custom_models(
    BASE_DIR="experiments",
    ANNOT_HASH="fba836ee1bdf4fda32004145ffe1eeb8d3c6b5f1",
    SPACY_ANNOT_MODEL="trf",
    SPACY_ANNOT_TYPE="docs",
):
    work_dir = Path(
        f"{BASE_DIR}/data_{ANNOT_HASH}/annotations_spacy_{SPACY_ANNOT_MODEL.upper()}"
    )

    files = sorted(
        [
            p.name
            for p in work_dir.glob(f"*{SPACY_ANNOT_TYPE}*")
            if not p.name.startswith(".")
        ]
    )

    annot_files = [f for f in files if f.startswith("ner_annotation")]
    run_dirs = [f for f in files if f.startswith("spacy")]

    trained_models = {
        d: sorted(list((work_dir / d / "spacy_model").glob("output*")))
        for d in run_dirs
    }

    return work_dir, annot_files, trained_models


def split_train_test_according_to_model(model):
    split = [p for p in model.parts if "TOT" in p][0]
    params = split.partition("TRSIZE")[-1].split("_RS")
    train_size = float(params[0].strip("_"))
    assert 0 < train_size < 1
    rs = int(params[1].split("_")[0])
    return train_size, rs


def load_trained_custom_model(
    trained_models=None, DATA_SPLIT_DIR=None, MODEL_INDEX=None
):
    MODEL_DIR = trained_models[DATA_SPLIT_DIR][MODEL_INDEX] / "model-best"

    train_size, rs = split_train_test_according_to_model(MODEL_DIR)

    return MODEL_DIR, train_size, rs


def get_entities_frequencies(data):
    freq = list(chain(*[[_[-1] for _ in item[1]["entities"]] for item in data]))
    _df1 = (
        pd.value_counts(freq, normalize=True)
        .map(lambda x: f"{x*100:.1f}%")
        .to_frame(name="freq")
    )
    _df2 = pd.value_counts(freq).to_frame(name="count")
    return _df1, _df2


def custom_spacy_styler(styler, title="Entity scores", color="green"):

    """
    Pandas styler for entity precision/recall/f1 scores for custom NER models
    """

    cm = sns.light_palette(color, n_colors=10, as_cmap=True)

    styler.format(precision=2)
    styler.set_properties(**{"border": "1px black solid !important"}).set_table_styles(
        [{"selector": "", "props": [("border", "1px black solid !important")]}]
    ).set_caption(title).set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("color", "black"),
                    ("font-size", "15px"),
                    ("text-align", "center"),
                    ("border-top", "1px black solid !important"),
                    ("border-left", "1px black solid !important"),
                    ("border-right", "1px black solid !important"),
                    # ('border-bottom','1px black solid !important')
                ],
            }
        ],
        overwrite=False,
    )  # Don't overwrite previous styles

    styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("font-size", "10pt"),
                    ("text-align", "center"),
                    ("border-style", "solid"),
                    ("border-width", "1px"),
                    # ('border-color', 'black')
                ],
            }
        ],
        overwrite=False,
    )
    styler.set_table_styles(
        [
            {
                "selector": "td",
                "props": [
                    ("font-size", "10pt"),
                    ("text-align", "center"),
                    ("border-style", "solid"),
                    ("border-width", "1px"),
                    # ('border-color', 'black')
                ],
            }
        ],
        overwrite=False,
    )
    styler.background_gradient(axis=0, low=0.0, high=1.0, cmap=cm)

    return styler


def default_spacy_styler(styler, title="f1-score by model type"):
    """
    Pandas styler for entity precision/recall/f1 scores for default spacy NER models
    """
    styler.format(precision=2)
    styler.set_properties(**{"border": "1px black solid !important"}).set_table_styles(
        [{"selector": "", "props": [("border", "1px black solid !important")]}]
    ).set_caption(title).set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("color", "black"),
                    ("font-size", "15px"),
                    ("text-align", "center"),
                    ("border-top", "1px black solid !important"),
                    ("border-left", "1px black solid !important"),
                    ("border-right", "1px black solid !important"),
                    # ('border-bottom','1px black solid !important')
                ],
            }
        ],
        overwrite=False,
    )  # Don't overwrite previous styles

    styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("font-size", "10pt"),
                    ("text-align", "center"),
                    ("border-style", "solid"),
                    ("border-width", "1px"),
                    # ('border-color', 'black')
                ],
            }
        ],
        overwrite=False,
    )
    styler.set_table_styles(
        [
            {
                "selector": "td",
                "props": [
                    ("font-size", "10pt"),
                    ("text-align", "center"),
                    ("border-style", "solid"),
                    ("border-width", "1px"),
                    # ('border-color', 'black')
                ],
            }
        ],
        overwrite=False,
    )
    styler.highlight_max(color="lightgreen", axis=1)
    styler.highlight_min(color="lightpink", axis=1)

    return styler
