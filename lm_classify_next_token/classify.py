from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import itertools
import math
from collections import defaultdict
import argparse

TOTAL_LINES = 72_019_562
N_CHOICES = 32
SYSTEM_PROMPT = """You are an expert at reading the metadata of academic articles and classifying them. The user will provide you some details about an academic article and you need to classify it into one of the following categories:

1. Audiovisual - A series of visual representations imparting an impression of motion when shown in succession. May or may not include sound. (May be used for films, video, etc.)
2. Award - An umbrella term for resources provided to individual(s) or organization(s) in support of research, academic output, or training, such as a specific instance of funding, grant, investment, sponsorship, scholarship, recognition, or non-monetary materials.
3. Book - A medium for recording information in the form of writing or images, typically composed of many pages bound together and protected by a cover.
4. BookChapter - One of the main divisions of a book.
5. Collection - An aggregation of resources, which may encompass collections of one resourceType as well as those of mixed types. A collection is described as a group; its parts may also be separately described. (A collection of samples, or various files making up a report)
6. ComputationalNotebook - A virtual notebook environment used for literate programming.
7. ConferencePaper - Article that is written with the goal of being accepted to a conference.
8. ConferenceProceeding - Collection of academic papers published in the context of an academic conference.
9. DataPaper - A factual and objective publication with a focused intent to identify and describe specific data, sets of data, or data collections to facilitate discoverability. (A data paper describes data provenance and methodologies used in the gathering, processing, organizing, and representing the data)
10. Dataset - Data encoded in a defined structure. (Data file or files)
11. Dissertation - A written essay, treatise, or thesis, especially one written by a candidate for the degree of Doctor of Philosophy.
12. Event - A non-persistent, time-based occurrence. (Descriptive information and/or content that is the basis for discovery of the purpose, location, duration, and responsible agents associated with an event such as a webcast or convention)
13. Image - A visual representation other than text. (Digitised or born digital images, drawings or photographs)
14. Instrument - A device, tool or apparatus used to obtain, measure and/or analyze data. (Note that this is meant to be the instrument instance, e.g., the individual physical device, not the digital description or design of an instrument.)
15. InteractiveResource - A resource requiring interaction from the user to be understood, executed, or experienced. (Training modules, files that require use of a viewer (e.g., Flash), or query/response portals)
16. Journal - A scholarly publication consisting of articles that is published regularly throughout the year.
17. JournalArticle - A written composition on a topic of interest, which forms a separate part of a journal.
18. Model - An abstract, conceptual, graphical, mathematical or visualization model that represents empirical objects, phenomena, or physical processes. (Modelled descriptions of, for example, different aspects of languages or a molecular biology reaction chain)
19. OutputManagementPlan - A formal document that outlines how research outputs are to be handled both during a research project and after the project is completed. (Includes data, software, and materials.)
20. PeerReview - Evaluation of scientific, academic, or professional work by others working in the same field.
21. PhysicalObject - A physical object or substance. (Artifacts, specimens, material samples, and features-of-interest of any size. Note that digital representations of physical objects should use one of the other resourceTypeGeneral values.)
22. Preprint - A version of a scholarly or scientific paper that precedes formal peer review and publication in a peer-reviewed scholarly or scientific journal.
23. Project - A planned endeavor or activity, frequently collaborative, intended to achieve a particular aim using allocated resources such as budget, time, and expertise. (This resource type represents the project and includes research projects and studies. For a project deliverable or description of a project, use the corresponding resource type for the output—e.g., for a project report, dissertation, or study registration, use the resourceTypeGeneral “Report”, “Dissertation”, or “StudyRegistration” instead.)
24. Report - A document that presents information in an organized format for a specific audience and purpose.
25. Service - An organized system of apparatus, appliances, staff, etc., for supplying some function(s) required by end users. (Data management service, or long-term preservation service)
26. Software - A computer program other than a computational notebook, in either source code (text) or compiled form. Use this type for general software components supporting scholarly research. Use the “ComputationalNotebook” value for virtual notebooks. (Software supporting scholarly research)
27. Sound - A resource primarily intended to be heard. (Audio recording)
28. Standard - Something established by authority, custom, or general consent as a model, example, or point of reference.
29. StudyRegistration - A detailed, time-stamped description of a research plan, often openly shared in a registry or published in a journal before the study is conducted to lend accountability and transparency in the hypothesis generating and testing process. (Includes pre-registrations, registered reports, and clinical trials. Study registrations are sometimes peer-reviewed and may include the hypothesis, expected results, study design, and/or analysis plan.)
30. Text - A resource consisting primarily of words for reading that is not covered by any other textual resource type in this list.
31. Workflow - A structured series of steps which can be executed to produce a final outcome, allowing users a means to specify and enact their work in a more reproducible manner. (Computational workflows involving sequential operations made on data by wrapped software and may be specified in a format belonging to a workflow management system, such as Taverna)
32. Other - Anything else

Pick only one of the above categories, whichever one fits best. Pay very careful attention to language and context. For instance,

* "book article", "book review", and "book series" should not be mapped to "Book"
* "notebook", "site notebook", etc. should not be mapped to "ComputationalNotebook"
* "event poster" and "event recording" should not be mapped to "Event"
* "journal artikel" etc. should not be mapped to "Journal" (should be "JournalArticle")
* "predigt" means "sermon" in German, so should not be mapped to "Preprint"
* "Program" can mean computer program ("Software") or a conference/symposium program ("Event")'
* Here is an example of a physical object: attributes.creators[0].name: GLIS Of The ITPGRFA; attributes.titles[0].title: Oryza sativa L. Os00129; attributes.publisher: {\"name\":\"CREA-Research Center for Cereals and Industrial Crops\"}; attributes.publicationYear: 2024; attributes.contributors[0].name: CREA-Research Center For Cereals And Industrial Crops; attributes.types.resourceType: PGRFA Material

Here are some clear examples to help you distinguish between commonly confused categories:

## Software vs Dataset

Example 1 (Software):
attributes.titles[0].title: FilteredPush/geo_ref_qc: Release version 2.0.0 of the geo_ref_qc library
attributes.publisher: Zenodo
Note: GitHub-style naming (username/repo), "Release version", and "library" are strong Software indicators.
Correct classification: 26 (Software)

Example 2 (Software):
attributes.titles[0].title: Code and data for publication "Dielectric profile at the Pt(111)/water interface"
attributes.publisher: Zenodo
Note: Even when "data" is mentioned, if "code" is explicitly stated, it's typically Software.
Correct classification: 26 (Software)

Example 3 (Dataset):
attributes.titles[0].title: Gene expression data from human lung tissue samples
attributes.publisher: Gene Expression Omnibus
Note: Pure data files without code components are Dataset.
Correct classification: 10 (Dataset)

## Book vs Image/Dataset

Example 4 (Book):
attributes.titles[0].title: Folding The Exhibition
attributes.publisher: Museu D'Art Contemporani De Barcelona
Note: Museum publications and exhibition catalogs are Books, not Images.
Correct classification: 3 (Book)

Example 5 (Book):
attributes.titles[0].title: Various Methods for Visualization and Size Determination of Gas Bubbles
attributes.publisher: University Press
Note: Academic books about visualization/imaging techniques are still Books, not Images.
Correct classification: 3 (Book)

## Patterns

- Software: Look for version numbers (v1.0, 2.0.0), GitHub patterns (user/repo), "release", "library", "code"
- Books: Check publisher type (museums, university presses), exhibition contexts, academic treatises
- Datasets: Pure data without code, data repositories, measurement/observation records

## Response

Respond only with the number of the category. Do not include any additional information. This is extremely important. Do not include any additional text, punctuation, or whitespace and end your message immediately after the number."""

CATEGORIES = {1: "Audiovisual", 2: "Award", 3: "Book", 4: "BookChapter", 5: "Collection", 6: "ComputationalNotebook", 7: "ConferencePaper", 8: "ConferenceProceeding", 9: "DataPaper", 10: "Dataset", 11: "Dissertation", 12: "Event", 13: "Image", 14: "Instrument", 15: "InteractiveResource", 16: "Journal", 17: "JournalArticle", 18: "Model", 19: "OutputManagementPlan", 20: "PeerReview", 21: "PhysicalObject", 22: "Preprint", 23: "Project", 24: "Report", 25: "Service", 26: "Software", 27: "Sound", 28: "Standard", 29: "StudyRegistration", 30: "Text", 31: "Workflow", 32: "Other"}

def parse_args():
    parser = argparse.ArgumentParser(description="Classify academic articles into categories.")

    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to use for classification.")
    parser.add_argument("--batch_size", type=int, default=1_000, help="Queue up this many articles before processing.")
    
    return parser.parse_args()

def process_batch(args, llm, tokenizer, article_batch, batch):
    # get the stringified description passed to the model
    article_desc = [x[1]['content'] for x in batch]

    # apply the chat template
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for messages in batch
    ]

    # sample three tokens at zero temperature
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        max_tokens=3,
        logprobs=1
    )
    outputs = llm.generate(prompts, sampling_params)

    # write the output to the output file
    with open(args.output_file, "a") as f:
        for metadata, desc, output in zip(article_batch, article_desc, outputs):
            metadata['prompt'] = desc

            # get the completion
            completion = output.outputs[0].text
            cumulative_logp = output.outputs[0].cumulative_logprob
            cumulative_prob = math.exp(cumulative_logp)
            
            # try and convert the completion to an integer
            category_pred = None
            try:
                completion_int = ''.join([x for x in completion if x.isdigit()])
                category_pred = int(completion_int)
                category_pred = CATEGORIES[category_pred]
            except:
                pass
            
            # write the output to the output file
            metadata['prediction'] = {
                'category': category_pred,
                'completion': completion,
                'probability': cumulative_prob
            }
            
            f.write(json.dumps(metadata) + "\n")

def main():
    args = parse_args()

    # load the model
    llm = LLM(
        args.model,
        rope_scaling = {"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768},
        max_model_len=32768*2
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # process the articles
    with open(args.input_file, "r") as f:
        article_batch = []
        batch = []

        for line in tqdm(f, total=TOTAL_LINES, desc="Assembling prompts"):
            # only keep rows with a resourceTypeGeneral so we can use it as the label
            metadata = json.loads(line)
            rtg = metadata.get('attributes.types.resourceTypeGeneral')
            if not rtg:
                continue

            # create the prompt
            article_desc = '\n'.join(
                f'{k}: {v}'
                for k, v in metadata.items()
                if k != 'attributes.types.resourceTypeGeneral'
            )
            prompt = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': article_desc}
            ]
            article_batch.append(metadata)
            batch.append(prompt)

            if len(batch) >= args.batch_size:
                process_batch(args, llm, tokenizer, article_batch, batch)
                batch = []
                article_batch = []
            
        if batch:
            process_batch(args, llm, tokenizer, article_batch, batch)

if __name__ == "__main__":
    main()
    