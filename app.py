import pandas as pd
import streamlit as st
import json
import uuid
from string_matching import StringMatching

st.set_page_config(layout="wide")

def read_json(file):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data = {}
    return data    


def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f)

def run(data_dir):
    if "load" not in st.session_state:
        uploaded_file = st.file_uploader(label="Choose a file", accept_multiple_files=False)
        if uploaded_file is None:
            return
        data = json.load(uploaded_file)           
        # data = read_json(f"{data_dir}/unlabelled.json")
        courses = read_json(f"{data_dir}/course.json")
        broad_fields = read_json(f"{data_dir}/broad_fos.json")
        narrow_fields = read_json(f"{data_dir}/narrow_fos.json")
        specific_fields = read_json(f"{data_dir}/specific_fos.json")
        annotated_samples = read_json(f"{data_dir}/annotations.json")
        course_code_map = {v: k for k,v in courses.items()}
        broad_field_code_map = {v: k for k,v in broad_fields.items()}
        narrow_field_code_map = {v: k for k,v in narrow_fields.items()}
        specific_field_code_map = {v: k for k,v in specific_fields.items()}
        sm = StringMatching(list(list(courses.values())))
        st.session_state["samples"] = {i: v for i,v in enumerate(data)}
        st.session_state["courses"] = courses
        st.session_state["broad_fields"] = broad_fields
        st.session_state["narrow_fields"] = narrow_fields
        st.session_state["specific_fields"] = specific_fields
        st.session_state["course_code_map"] = course_code_map
        st.session_state["broad_field_code_map"] = broad_field_code_map
        st.session_state["narrow_field_code_map"] = narrow_field_code_map
        st.session_state["specific_field_code_map"] = specific_field_code_map
        st.session_state["annotated_samples"] = annotated_samples
        st.session_state["sample_index"] = 0
        st.session_state["sm"] = sm
        st.session_state["load"] = 1
        st.experimental_rerun()
    else:
        data = st.session_state["samples"]
        broad_fields = st.session_state["broad_fields"]
        narrow_fields = st.session_state["narrow_fields"]
        specific_fields = st.session_state["specific_fields"]
        course_code_map = st.session_state["course_code_map"]
        broad_field_code_map = st.session_state["broad_field_code_map"]
        narrow_field_code_map = st.session_state["narrow_field_code_map"]
        specific_field_code_map = st.session_state["specific_field_code_map"]
        sm = st.session_state["sm"]


    def get_lower_cats(code, lvl):
        if lvl == 2:
            x = narrow_fields
            match_num = 1
        elif lvl == 3:
            x = specific_fields
            match_num = 2
        x_filtered = [j for i,j in x.items() if i.split('-')[:match_num]==code.split('-')]
        return x_filtered


    def parse_course_code(code):
        code_split = code.split('-')
        broad_code = code_split[0]
        narrow_code = '-'.join(code_split[:2])
        specific_code = '-'.join(code_split[:3])
        broad_field = broad_fields[broad_code]
        narrow_field = narrow_fields[narrow_code]
        specific_field = specific_fields[specific_code]
        return (broad_code, broad_field), (narrow_code, narrow_field), (specific_code, specific_field) 


    def predict_match(sample):
        res = sm.query(sample)
        return res[0]['match'][0]


    def get_best_match(sample):
        pred_course = predict_match(sample)
        broad_field = "education"
        narrow_field = "education"
        specific_field = "education science"
        if pred_course:
            pred_code = course_code_map[pred_course]
            (_, broad_field), (_, narrow_field), (_, specific_field) = parse_course_code(pred_code)
        return broad_field, narrow_field, specific_field, pred_course


    def next_sample():
        sample_index = st.session_state["sample_index"]
        if sample_index < len(st.session_state["samples"]) - 1:
            st.session_state["sample_index"] += 1
            # uncheck ignore checkbox
            st.session_state["ignore_cb"] = False


    def previous_sample():
        sample_index = st.session_state["sample_index"]
        if sample_index > 0:
            st.session_state["sample_index"] -= 1
            # uncheck ignore checkbox
            st.session_state["ignore_cb"] = False   

    def goto_sample():
        st.session_state["sample_index"] = st.session_state["goto_id"] - 1
        # uncheck ignore checkbox
        st.session_state["ignore_cb"] = False

    def add_sample():
        st.session_state["samples"][len(data)] = f"Sample {str(uuid.uuid4())}"   
        st.session_state["sample_index"] = len(data) - 1
        st.session_state["ignore_cb"] = False
        st.experimental_rerun()

    def annotate(sample, edited_sample, broad_label, narrow_label, specific_label, ignore_sample):
        st.session_state["annotated_samples"][sample] = {
                "edited_sample": edited_sample,
                "broad_label": broad_label,
                "narrow_label": narrow_label,
                "specific_label": specific_label,
                "ignore_sample": ignore_sample}
        # if len(st.session_state["samples"]) == len(st.session_state["annotated_samples"]):
        #     save_json(st.session_state["annotated_samples"], f"{data_dir}/annotations.json")
        #     st.info("All annotations saved")
    


    # sidebar: show status
    n_samples = len(st.session_state["samples"])
    n_annotation_samples = len(st.session_state["annotated_samples"])
    
    # download button
    # with open("annotations.json", "w") as f:
    st.sidebar.download_button(
        label="Download data as JSON",
        data=json.dumps(st.session_state["annotated_samples"]),
        file_name='annotations.txt',
        mime='text/plain',
    )


    # current sample
    sample_id = st.session_state['sample_index']
    sample = data[sample_id]
    broad_field_match, narrow_field_match, specific_field_match, course_match = get_best_match(sample)
    ignore_sample = False
    st.sidebar.write("Total samples:", n_samples)
    st.sidebar.write("Total annotated samples:", n_annotation_samples)
    st.sidebar.write("Remaining samples:", n_samples - n_annotation_samples)

    col_s1, col_s2, col_s3 = st.sidebar.columns(3)
    with col_s1:
        st.button(label="Previous", on_click=previous_sample)
    with col_s2:
        st.button(label="Next", on_click=next_sample)

    # save annotations
    # if st.sidebar.button(label="Save annotations", help="save annotations done so far"):
    #     save_json(st.session_state["annotated_samples"], f"{data_dir}/annotations.json")
    #     st.info("Saved annotations done so far")

    # add sample
    if st.sidebar.button(label="Add Sample", help=f"add an extra sample #{n_samples+1} to annotate"):
        add_sample()

    # goto sample number
    st.sidebar.number_input(label='Go to',
                            key="goto_id",
                            value=st.session_state["sample_index"]+1,
                            min_value=1,
                            max_value=n_samples,
                            step=1,
                            on_change=goto_sample)
                                
    

    # status grid
    n_cols = 5
    n_rows = n_samples//n_cols + 1
    cp_id = 1
    for _ in range(n_rows):
        with st.sidebar.container():
            cols = st.columns(n_cols)
            for col in cols:
                # check if sample is annotated
                if cp_id - 1 == sample_id:
                    status_color = "#00a6f9" # blue, current sample
                elif data[cp_id-1] in st.session_state["annotated_samples"]:
                    status_color = '#00f900' # green, annotated
                else:
                    status_color = '#f90000' # red, not annotated
                col.color_picker(str(cp_id), status_color, key=f"sc_{cp_id}", disabled=True)
                if cp_id == n_samples:
                    break
                cp_id += 1
            else:
                continue
            break

    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader(sample)
        if sample in st.session_state["annotated_samples"]:
            sample_val = st.session_state["annotated_samples"][sample]["edited_sample"]
            status_color = '#00f900' # green, annotated
        else:
            sample_val = sample
            status_color = '#f90000' #red, not annotated
        st.color_picker('status', status_color, label_visibility="collapsed", disabled=True)
        edited_sample = st.text_input("Edit course name", value=sample_val)
        ignore_sample = st.checkbox("Ignore", key="ignore_cb", value=False)
    with col_2:
        st.write("**Best string-match**")
        st.write(f"**Course** : {course_match}")
        st.write(f"**Specific FOS** : {specific_field_match}")
        st.write(f"**Narrow FOS** : {narrow_field_match}")
        st.write(f"**Broad FOS** : {broad_field_match}")

    broad_label_options = list(broad_fields.values())
    col_3, col_4, col_5 = st.columns(3)  

    with col_3:
        default_index = broad_label_options.index(broad_field_match)
        broad_label = st.radio(label='Broad FOS',
                  options=broad_label_options,
                  index=default_index)

    broad_label_id = broad_field_code_map[broad_label]
    narrow_field_cats = get_lower_cats(broad_label_id, 2)
    
    with col_4:
        try:
            default_index = narrow_field_cats.index(narrow_field_match)
        except ValueError:
            default_index = 0
        narrow_label = st.radio(label='Narrow FOS',
                        options=narrow_field_cats,
                        index=default_index)

    narrow_label_id = narrow_field_code_map[narrow_label]
    specific_field_cats = get_lower_cats(narrow_label_id, 3)
    
    with col_5:
        try:
            default_index = specific_field_cats.index(specific_field_match)
        except ValueError:
            default_index = 0
        specific_label = st.radio(label='Specific FOS',
                            options=specific_field_cats,
                            index=default_index)
    specific_label_id = specific_field_code_map[specific_label]

    # tag button
    with col_s3:
        if st.button(label="Tag", help="tag current sample with selected labels"):
                annotate(sample, edited_sample, broad_label_id, narrow_label_id, specific_label_id, ignore_sample)
                st.experimental_rerun()

    
if __name__ == "__main__":
    run("data_dir")