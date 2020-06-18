# Version 1.0
# Python 3.6
# Install tqdm for tracking progress

from tqdm import tqdm
import json
import sys
import os
import io
import codecs

def parse_sim_file(filename):
    """
    Reads the deduplicated documents file and stores the
    duplicate passage ids into a dictionary
    """
    # counter = 0
    sim_dict = {}
    lines = open(filename).readlines()
    for line in lines:
        data = line.strip().split(':')
        # if counter == 144685:

        # counter=counter+1
        if (len(data)==2):
            if len(data[1]) > 0:
                sim_docs = data[-1].split(',')
                for docs in sim_docs:
                    sim_dict[docs] = 1

    return sim_dict

def write_to_file(idx, text):
    # Writes the passage contents in trecweb format
    # content = (u'<DOC>\n')
    # content += (u'<DOCNO>')
    content = (str(idx))+ (u'\t')
    # content += (u'</DOCNO>\n')
    # content += (u'<DOCHDR>\n')
    # content += (u'\n')
    # content += (u'</DOCHDR>\n')
    # content += (u'<HTML>\n')
    # content += (u'<BODY>\n')
    content += (text)
    content += (u'\n')
    # content += (u'</BODY>\n')
    # content += (u'</HTML>\n')
    # content += (u'</DOC>\n')
    fp.write(content)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("USAGE: python marco_trecweb.py path_to_collection.tsv path_of_dumpdir similarity_file")
        exit(0)

    marco_file = sys.argv[1]
    dump_dir = sys.argv[2]
    sim_file = sys.argv[3]

    # Create the directory (for dumping files) if it doesn't exists
    # if not os.path.exists(dump_dir):
    #     os.mkdir(dump_dir)

    print("Loading similarity file.")
    sim_dict = parse_sim_file(sim_file)

    input_file = os.path.basename(marco_file)

    print("Starting processing.")
    print("Output directory: " + dump_dir)
    dumper_file = os.path.join(dump_dir, input_file + '.xml')
    print("Writing output to: " + dumper_file)
    fp = codecs.open(dumper_file, 'w', 'utf-8')

    # Read the ranking collections file
    with io.open(marco_file, "r", encoding="utf-8") as input:

        for line in tqdm(input, total=8841823):

            # Split to get the original id and text
            idx, text = line.strip().split('\t', 1)
            idx = 'MARCO_' + idx

            # if the id is a duplicate, don't add it
            if idx in sim_dict:

                continue

            # Create a trecweb entry for a passage
            write_to_file(idx, text)

    input.close()
    fp.close()