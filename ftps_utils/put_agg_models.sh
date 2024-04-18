#!/bin/bash

# FTPS server details

if [[ "$1" == "bp" ]]; then
    echo "Loading BP FTPS credentials..."
    source bp_credentials.sh
elif [[ "$1" == "petrobras" ]]; then
    echo "Loading Petrobras FTPS credentials..."
    source petrobras_credentials.sh
else
    echo "Invalid argument"

# File to store the previous file listing
PREV_LIST_FILE="previous_server_model_list.txt"
CURRENT_LIST_FILE="current_server_model_list.txt"

# Function to retrieve file listing from FTPS server
get_file_listing() {
    ls server_models/ | awk '{print $NF}' > "$CURRENT_LIST_FILE"
}

# Function to upload new folders from FTPS server
upload_server_models() {
    diff --unchanged-line-format="" "$PREV_LIST_FILE" "$CURRENT_LIST_FILE" | grep -v '^$' | while read -r model; do
        echo "Uploading new server model: $model"
        lftp -u "$USERNAME","$PASSWORD" "ftps://$HOST$REMOTE_DIR" -e "put --only-newer server_models/$model -o /server_models/$(model); exit"
    done
}

# Check if previous file listing exists
if [ -f "$PREV_LIST_FILE" ]; then
    # Get current file listing
    get_file_listing

    # Compare file listings
    diff "$PREV_LIST_FILE" "$CURRENT_LIST_FILE" > /dev/null
    if [ $? -eq 0 ]; then
        echo "No changes detected."
    else
        echo "Changes detected! Downloading new files..."
	
	upload_server_models
        # Perform actions if changes are detected
        # TODO: run Python client with the new round
        # python client.py ...
    fi
else
    echo "First run. Retrieving file listing..."
    get_file_listing
    echo "File listing saved."
fi

mv "$CURRENT_LIST_FILE" "$PREV_LIST_FILE"
