# Version 1.0
import os
import sys
import codecs
from trec_car.read_data import *

def writer(p, fp):

    """
    Writes each paragraph in the trecweb format
    """
    # Get the paragraph id and text
    para_id = 'CAR_' + str(p.para_id)
    text = p.get_text()

    # content = (u'<DOC>\n')
    # content += (u'<DOCNO>')
    content = (para_id) + (u'\t')
    # content += (u'</DOCNO>\n')
    # content += (u'<DOCHDR>\n')
    # content += (u'\n')
    # content += (u'</DOCHDR>\n')
    # content += (u'<BODY>\n')
    content += (text)
    # content += (u'\n</BODY>\n')
    # content += (u'</DOC>\n')
    content += (u'\n')
    fp.write(content)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python car_treweb.py dedup.articles-paragraphs.cbor DUMP_DIR")
        exit(0)

    filename = sys.argv[1]
    dump_dir = sys.argv[2]

    input_file = os.path.basename(filename)

    dumper_file = os.path.join(dump_dir, input_file + '.xml')
    print("Writing output to: " + dumper_file)
    fp = codecs.open(dumper_file, 'w', 'utf-8')
    print("Starting processing.")
    print("Output directory: " + dump_dir)

    # Reads the file and iterates over paragraphs
    total = 0
    print("Reading ", filename)
    with open(filename, 'rb') as rp:
        for p in iter_paragraphs(rp):

            # Write to file
            writer(p, fp)
            total += 1
    print("Total paras written = ", total)
    print("Closing File")

    rp.close()
    fp.close()

