#!/bin/bash

# FTPS server details
if [[ "$1" == "bp" ]]; then
    echo "Loading BP FTPS credentials..."
    source BP_credentials.sh
elif [[ "$1" == "petrobras" ]]; then
    echo "Loading Petrobras FTPS credentials..."
    source Petrobras_credentials.sh
else
    echo "Invalid argument"
fi


# File to store the previous file listing
PREV_LIST_FILE="$1_previous_file_list.txt"
CURRENT_LIST_FILE="$1_current_file_list.txt"


# Function to retrieve file listing from FTPS server
get_file_listing() {
    lftp -u "$USERNAME","$PASSWORD" "ftps://$HOST$REMOTE_DIR" -e "set ftp:ssl-force true; ls client_models; exit" | awk '{print $NF}' > "$CURRENT_LIST_FILE"
}

# Function to download new folders from FTPS server
download_new_folders() {
    diff --unchanged-line-format="" "$PREV_LIST_FILE" "$CURRENT_LIST_FILE" | grep -v '^$' | while read -r folder; do
        echo "Downloading new folder: $folder to $STOCKYARD/fl/client_models/$1/$folder"
        lftp -u "$USERNAME","$PASSWORD" "ftps://$HOST$REMOTE_DIR" -e "cd client_models; get $folder; exit"
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
	
	download_new_folders $1
	mv *.tar* $STOCKYARD/fl/client_models/petrobras
	tar -xvf $STOCKYARD/fl/client_models/petrobras/*.tar*
	#rm $STOCKYARD/fl/client_models/petrobras/*.tar*
	for d in $STOCKYARD/fl/client_models/petrobras/round_*; do cp $STOCKYARD/fl/client_models/petrobras/training_size.txt "$d"; done
	
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
