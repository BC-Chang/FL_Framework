#!/bin/bash

# FTPS server details
HOST=
USERNAME=
PASSWORD=
REMOTE_DIR=

# File to store the previous file listing
PREV_LIST_FILE="previous_file_list.txt"
CURRENT_LIST_FILE="current_file_list.txt"

# Function to retrieve file listing from FTPS server
get_file_listing() {
    lftp -u "$USERNAME","$PASSWORD" "ftps://$HOST$REMOTE_DIR" -e "ls; exit" | awk '{print $NF}' > "$CURRENT_LIST_FILE"
}

# Function to download new folders from FTPS server
download_new_folders() {
    diff --unchanged-line-format="" "$PREV_LIST_FILE" "$CURRENT_LIST_FILE" | grep -v '^$' | while read -r folder; do
        echo "Downloading new folder: $folder"
        lftp -u "$USERNAME","$PASSWORD" "ftps://$HOST$REMOTE_DIR" -e "mirror --only-newer $folder ./downloaded_folders/$folder; exit"
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
	
	download_new_folders
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
