# Version 1.0
# Python 3

import json
import sys
import os
import codecs
import re
from tqdm import tqdm

def write_to_file(fp, data, i):

    """ 
    Converts all the available paragraphs into the trecweb format
    and writes it to a file. Ids to each paragraphs are assigned
    in a sequential order. If an id belongs to a duplicate one,
    it is not added.

    Args:
        fp: file pointer for writing the trecweb docs
        data: json parsed line containing all the paragraphs
        i: (optional) index of parsed data
   
    """

    main_idx = 'WAPO_' + str(data['id'])
    main_url = data['article_url']

    paras = data['contents']
    counter = 0
    for idx, each in enumerate(paras):
        if each == None:
            continue
        elif 'subtype' in each:
            if each['subtype'] == 'paragraph':
                body =  each['content']
                counter += 1
                paraidx = main_idx + '-' + str(counter)
                if paraidx in dup_dict:
                    import pdb; pdb.set_trace()  # breakpoint fa31cd7b //
                    
                    continue
                content = u'<DOC>\n'
                content += u'<DOCNO>'
                content += str(paraidx)
                content += u'</DOCNO>\n'
                content += u'<DOCHDR>\n'
                content += main_url
                content += u'\n</DOCHDR>\n'
                content += u'<HTML>\n'
                content += u'<BODY>\n'
                content += body
                content += '\n'
                content += u'</BODY>\n'
                content += '</HTML>\n'
                content += '</DOC>\n'
                fp.write(content)

def parse(file_path, dumper):

    """ 
    Iterates over each file in the WAPO dataset.
    Each file is opened, its contents are read,
    and the paragraphs are dumped in the trecweb
    format.
    """

    if not os.path.exists(dumper):
        os.mkdir(dumper)

    for file in os.listdir(file_path):
        dumper_file = os.path.join(dumper, file + '.xml')
        fp = codecs.open(dumper_file, 'w', 'utf-8')
        file = os.path.join(file_path, file)
        print("Opening ", file)
        lines = open(file, 'r').readlines()
        print("Read ", file)
        tl = len(lines)
        for i, data in tqdm(enumerate(lines), total=tl):
            data1 = data.strip()
            data1 = json.loads(data1)
            write_to_file(fp, data1, i)
        fp.close()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python wapo_clean_parser.py DATAPATH DUMP_PATH DUPLICATE_FILE")
        print("Example: python wapo_clean_parser.py ../wapo_path ../wapo_dump_dir duplicate.list")
        exit(0)


    dups_file = sys.argv[3]
    dumper = sys.argv[2] 
    data_path = sys.argv[1]

    # Creates a dict for duplicates for easy access
    dup_dict = {}
    data_dups = open(dups_file).readlines()
    for each in data_dups:
        idxs = each.strip().split(':')
        if len(idxs[-1]) > 0:
            all_idxs = idxs[-1].split(',')
            for every in all_idxs:
                dup_dict[every] = 1

    # Parse all the files in the directory
    parse(data_path, dumper, dup_dict)
