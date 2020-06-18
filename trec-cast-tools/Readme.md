# Tools and scripts for TREC CAsT

## Topic file processing
Code for processing the topic files in Java (all three formats) is available in:
src/main/java

A maven file is provide for building the code. 

## Convert various collections to TREC Web format. 
To run the parser for CAR corpus V2:<br/>
    Setup trec car tools from https://github.com/TREMA-UNH/trec-car-tools<br/>
    Run: python car_trecweb.py PATH_TO_CBORFILE OUTPUT_DIRECTORY

To run the parser for Washington Post:
    Run: python wapo_trecweb.py DATAPATH OUTPUT_DIRECTORY<br/>
    Here DATAPATH is the directory containing the json files of Washington Post data,
    and OUTPUT_DIRECTORY is the name of the directory where you want to store the converted files

To run the parser for MSMARCO:
    Install tqdm if you dont already have it<br/>
    Run: python marco_trecweb.py path_to_collection.tsv OUTPUT_DIRECTORY DUPLICATE_FILE<br/>
    Here path_to_collection.tsv is the tab seperated MARCO data,
    OUTPUT_DIRECTORY is the location of the directory where you want to store the converted files,
    and DUPLICATE_FILE is the file containing the list of deduplicated documents.

NOTE: The scripts have been tested with Python 3.6 (but anything >= 3.5 should work). 
