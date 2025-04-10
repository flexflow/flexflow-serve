import streamlit as st
import requests
import os, json
from huggingface_hub import model_info
import threading
import time
import pandas as pd
from multiprocessing import Process, Pipe

# App title
st.set_page_config(page_title="🚀💻 FlexLLM Server", layout="wide")

# FastAPI server URL
CHAT_URL = "http://localhost:8080/chat/completions/"  # Adjust the port if necessary
FINETUNE_URL = "http://localhost:8080/finetuning/"
REGISTER_ADAPTER_URL = "http://localhost:8080/register_adapter/"
GET_DATASET_CONFIGS_URL = "http://localhost:8080/get_dataset_configs/"
GET_DATASET_SPLITS_URL = "http://localhost:8080/get_dataset_splits/"
GET_DATASET_COLUMNS_URL = "http://localhost:8080/get_dataset_columns/"
UPLOAD_PEFT_MODEL_URL = "http://localhost:8080/upload_peft_model/"
PROGRESS_URL = "http://localhost:8080/training_progress/"

# Initialize session state variables
if 'added_adapters' not in st.session_state:
    st.session_state.added_adapters = []

if 'adapter' not in st.session_state:
    st.session_state.adapter = None
    st.session_state.peft_model_id = None

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def check_model_availability(model_name):
    try:
        info = model_info(model_name)
        return True
    except Exception:
        return False

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function for generating LLaMA2 response
def generate_llama3_response(prompt_input):
    system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are positive in nature."
    
    print("---Front: here is inference request data----")
    print(prompt)

    # Send request to FastAPI server
    response = requests.post(
        CHAT_URL,
        json={
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "decoding_method": decoding_method,
            "peft_model_id": st.session_state.peft_model_id if st.session_state.peft_model_id else None,
            "messages": [{"role": "system", "content": system_prompt}] + st.session_state.messages + [{"role": "user", "content": prompt_input}]
        }
    )

    result = response.json()
    if response.status_code == 200:
        if result["status"] == "success":
            return result["response"]
    else:
        return f"{result['detail']}"

finetune_result = None
st.session_state.start_finetune = None
st.session_state.request_data = None
st.session_state.peft_model_id = None

# Sidebar
with st.sidebar:
    st.title('🚀 FlexLLM Server')
    page = st.radio("Choose a page", ["Chat", "Finetune"])
    peft_model_id = None
    if page == "Chat":
        st.header('🦙 Llama Chatbot')
        # st.success('Using local FastAPI server', icon='✅')
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        st.subheader('Generation parameters')
        max_length = st.sidebar.slider('Max generation length', min_value=64, max_value=2048, value=256, step=8)
        # selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'], key='selected_model')
        decoding_method = st.sidebar.selectbox('Decoding method', ['Greedy decoding (default)', 'Sampling'], key='decoding_method')
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01, disabled=decoding_method == 'Greedy decoding (default)')
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, disabled=decoding_method == 'Greedy decoding (default)')

        # Single LoRA Adapter
        st.subheader("LoRA Adapter (optional)")
        placeholder_text = "Enter the Huggingface PEFT model ID" if not st.session_state.get("adapter") else f"Currently registered: {st.session_state.adapter}"
        peft_model_name = st.text_input("Add a LoRA Adapter", placeholder=placeholder_text)
        # Button to register the adapter
        if st.button("Register Adapter"):
            if peft_model_name:
                with st.spinner("Registering LoRA adapter..."):
                    response = requests.post(REGISTER_ADAPTER_URL, params={"peft_model_name": peft_model_name})

                result = response.json()
                if response.status_code == 200:
                    if result["status"] == "success":
                        st.session_state.adapter = peft_model_name
                        st.session_state.peft_model_id = result["peft_model_id"]
                        st.success(f"Successfully registered adapter: {peft_model_name}")
                    else:
                        st.error(f"Failed to register adapter: {result['detail']}")
                else:
                    st.error(f"Failed to register adapter: {result['detail']}")
            else:
                st.warning("Please enter a PEFT model ID.")
                
        # Button to remove current adapter
        if st.button("Remove Current Adapters"):
            st.session_state.adapter = None
            st.session_state.peft_model_id = None
            st.success("Adapter has been removed.")

        # Display current adapter
        st.markdown("**Current Adapter:**")
        if st.session_state.adapter:
            st.write(f"- {st.session_state.adapter}")
        else:
            st.write("No adapter registered.")

        # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
    elif page == "Finetune":
        st.header("🏋️‍♂️ LoRA Finetuning")
        
        # Hugging Face token input
        if 'hf_token' in st.session_state.keys():
            st.success('HF token already provided!', icon='✅')
            hf_token = st.session_state.hf_token
        else:
            hf_token = st.text_input('Enter your Hugging Face token:', type='password')
            if not (hf_token.startswith('hf_') and len(hf_token)==37):
                st.warning('please enter a valid token', icon='⚠️')
            else:
                st.success('Proceed to finetuning your model!', icon='👉')
                st.session_state.hf_token = hf_token
        
        # Upload PEFT model information
        peft_model_name = st.text_input(
            "Enter the PEFT model name to upload to Hugging Face:",
            help="The name of the PEFT model should start with the username associated with the provided HF token, followed by '/'ß. E.g. 'username/peft-base-uncased'"
        )
        private = st.checkbox("Upload as a private model")
        
        # Dataset selection
        dataset_option = st.radio("Choose dataset source:", ["Upload JSON", "Hugging Face Dataset"])
        
        if dataset_option == "Upload JSON":
            uploaded_file = st.file_uploader("Upload JSON dataset", type="json")
            if uploaded_file is not None:
                dataset = json.load(uploaded_file)
                st.success("Dataset uploaded successfully!")
        else:
            if "selected_dataset" not in st.session_state:
                st.session_state.selected_dataset = None
                st.session_state.selected_config = None
                st.session_state.selected_split = None

            dataset_name = st.text_input(
                "Enter Hugging Face dataset name:",
                help="The dataset name should follow the format 'username/dataset-name'"
            )

            # Initialize placeholders with disabled state
            config_placeholder = st.empty()
            split_placeholder = st.empty()
            column_placeholder = st.empty()
            # Disable UI elements initially
            config_select = config_placeholder.selectbox("Select config name:", ["..."], disabled=True)
            split_select = split_placeholder.selectbox("Select dataset split:", ["..."], disabled=True)
            column_select = column_placeholder.selectbox("Select column for finetuning:", ["..."], disabled=True)

            if dataset_name and dataset_name != st.session_state.selected_dataset:
                st.session_state.selected_dataset = dataset_name
                # Get config names
                response = requests.get(f"{GET_DATASET_CONFIGS_URL}?dataset_name={dataset_name}")
                result = response.json()
                st.session_state.selected_config = None # Reset selected_config
                if response.status_code == 200: # Get config_names
                    st.session_state.config_names = result["config_names"]
                else:
                    st.session_state.config_names = None
                    config_select = config_placeholder.selectbox("Select config name:", ["Error"], disabled=True)
                    st.error(f"{result['detail']}")
            selected_config = []
            if "config_names" in st.session_state and st.session_state["config_names"]:
                selected_config = config_placeholder.selectbox("Select config name:", st.session_state.config_names, disabled=False)
            
            if dataset_name and selected_config != st.session_state.selected_config:
                st.session_state.selected_config = selected_config
                split_url = f"{GET_DATASET_SPLITS_URL}?dataset_name={dataset_name}"
                if selected_config:  # Ensure config_name is added only if it's not None
                    split_url += f"&config_name={selected_config}"
                # Get splits
                response = requests.get(split_url)
                result = response.json()
                st.session_state.selected_split = None 
                if response.status_code == 200:
                    st.session_state.splits = result["splits"]
                else:
                    st.session_state.splits = None
                    split_select = split_placeholder.selectbox("Select dataset split:", ["Error"], disabled=True)
                    st.error(f"{result['detail']}")
            selected_split = []
            if "splits" in st.session_state and st.session_state["splits"]:
                selected_split = split_placeholder.selectbox("Select dataset split:", st.session_state.splits, disabled=False) if "splits" in st.session_state else None

            if dataset_name and selected_split != st.session_state.selected_split:
                st.session_state.selected_split = selected_split
                columns_url = f"{GET_DATASET_COLUMNS_URL}?dataset_name={dataset_name}&split={selected_split}"
                if selected_config:  # Add only if selected_config is not None
                    columns_url += f"&config_name={selected_config}"
                # Get available columns
                response = requests.get(columns_url)
                result = response.json()
                if response.status_code == 200:
                    st.session_state.columns = result["columns"]
                else:
                    st.session_state.columns = None
                    column_select = column_placeholder.selectbox("Select column for finetuning:", ["Error"], disabled=True)
                    st.error(f"Failed to get dataset columns: {result['detail']}")
            selected_column = []
            if "columns" in st.session_state and st.session_state["columns"]:
                selected_column = column_placeholder.selectbox("Select column for finetuning:", st.session_state.columns, disabled=False) if "columns" in st.session_state else None

        # Finetuning parameters
        st.subheader("Finetuning parameters")
        lora_rank = st.number_input("LoRA rank", min_value=2, max_value=64, value=16, step=2)
        lora_alpha = st.number_input("LoRA alpha", min_value=2, max_value=64, value=16, step=2)
        target_modules = st.multiselect("Target modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"], default=["down_proj"])
        learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=1.0, value=0.001, step=0.0001, format="%.4f")
        optimizer_type = st.selectbox("Optimizer type", ["SGD", "Adam", "AdamW", "Adagrad", "Adadelta", "Adamax", "RMSprop"])
        momentum = st.number_input("Momentum", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        nesterov = st.checkbox("Nesterov")
        max_training_epochs = st.number_input("Max training epochs", min_value=1, max_value=5000, value=10, step=50)

        # Start finetuning button
        if st.button("Start Finetuning"):
            if not hf_token:
                st.error("Please enter your Hugging Face token.")
            elif dataset_option == "Upload JSON" and uploaded_file is None:
                st.error("Please upload a JSON dataset.")
            elif dataset_option == "Hugging Face Dataset" and (not dataset_name or not selected_split or not selected_column):
                st.error("Please enter all Hugging Face dataset information.")
            else:
                # Prepare the request data
                st.session_state.request_data = {
                    "token": hf_token,
                    "peft_model_id": peft_model_name,
                    "dataset_option": dataset_option,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "target_modules": target_modules,
                    "learning_rate": learning_rate,
                    "optimizer_type": optimizer_type,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "nesterov": nesterov,
                    "max_training_epochs": max_training_epochs,
                }

                if dataset_option == "Upload JSON":
                    st.session_state.request_data["dataset"] = dataset
                else:
                    st.session_state.request_data["dataset_name"] = dataset_name
                    st.session_state.request_data["config_name"] = selected_config
                    st.session_state.request_data["selected_split"] = selected_split
                    st.session_state.request_data["selected_column"] = selected_column
                
                st.session_state.upload_request_data = {
                    "token": hf_token,
                    "peft_model_id": peft_model_name,
                    "upload_peft_model_id": peft_model_name,
                    "private": private
                }
                st.session_state.peft_model_id = peft_model_name

                print("---Front: here is finetune request data----")
                print(st.session_state.request_data)

                st.session_state.start_finetune = True

if page == "Chat":
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Running..."):
                response = generate_llama3_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
elif page == "Finetune":
    # st.subheader("📈 Training Progress")
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    # Send finetuning request to FastAPI server
    # Send the finetuning request in a separate thread
    if st.session_state.start_finetune:
        st.subheader("📈 Training Progress")
        def run_finetune_request():
            finetune_response = requests.post(FINETUNE_URL, json=st.session_state.request_data)
            st.session_state.finetune_response = finetune_response

        t = threading.Thread(target=run_finetune_request)
        add_script_run_ctx(t, get_script_run_ctx())
        t.start()

        time.sleep(3)
        if "finetune_response" in st.session_state and st.session_state.finetune_response.status_code != 200:
            st.error(f"{st.session_state.finetune_response.json().get('detail', 'Unknown error occurred.')}")
            # 409
        
        # 2. Meanwhile: poll training progress and update UI
        progress_placeholder = st.empty()
        chart_placeholder = st.empty()
        status_placeholder = st.empty()

        while True:
            try:
                response = requests.get(PROGRESS_URL)
                st.session_state.progress = response.json()
            except:
                st.warning("Failed to get training progress.")
                continue

            current_epoch = st.session_state.progress["current_epoch"]
            max_epochs = st.session_state.progress.get("max_epochs", 1)
            loss_history = st.session_state.progress.get("loss_history", [])

            # Progress bar
            progress_pct = current_epoch / max_epochs if max_epochs else 0
            progress_placeholder.progress(progress_pct, text=f"Epoch {current_epoch}/{max_epochs}")

            # Live loss chart
            if loss_history:
                loss_df = pd.DataFrame({"Loss": loss_history})
                loss_df.index += 1  # Epochs start from 1
                chart_placeholder.line_chart(loss_df)
                status_placeholder.info(f"Latest Loss: {loss_history[-1]:.4f}")

            if st.session_state.progress["status"] == "done":
                break
            
            time.sleep(2)  # update every 2 seconds

        while not "finetune_response" in st.session_state:
            time.sleep(2)

        finetune_result = st.session_state.finetune_response.json()
        if st.session_state.finetune_response.status_code == 200:
            st.success("Finetuning completed successfully!")

            # Start uploading model to hf
            with st.spinner("Uploading fine-tuned model to Hugging Face..."):
                upload_response = requests.post(UPLOAD_PEFT_MODEL_URL, json=st.session_state.upload_request_data)

            upload_result = upload_response.json()
            if upload_response.status_code == 200:
                st.success(f"{peft_model_id} Model uploaded successfully to Hugging Face!")
            else:
                st.error(f"Upload failed: {upload_result.get('detail', 'Unkifnown error occurred.')}")
        elif st.session_state.finetune_response.status_code != 409:
            st.error(f"Finetuning failed: {finetune_result.get('detail', 'Unknown error occurred.')}")

    # Print out the number of entries
    if finetune_result and finetune_result.get("total_entries") and finetune_result.get("remaining_entries"):
        st.write(f"Dataset loaded: {finetune_result['total_entries']} entries found.")
        st.write(f"{finetune_result['remaining_entries']} entries remaining after filtering with max sequence length.")
    # elif not "progress" in st.session_state:
    else:
        st.write("Use the sidebar to configure and start finetuning.")
