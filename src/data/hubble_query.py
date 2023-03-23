import os
import re

import argparse
from pathing import QUERY_PATH
from astroquery.esa.hubble import ESAHubble

def ends_with_pattern(string):
    pattern = r"_\d+$"  # define a regular expression pattern
    if re.search(pattern, string):
        return True
    else:
        return False
    
def getHubbleQuery():
    return hubbler.query_criteria(calibration_level = args.calibration_level,
                                  data_product_type = args.data_product_type,
                                  intent=args.intent,
                                  obs_collection=args.obs_collection,
                                  instrument_name = args.instrument_name,
                                  filters = args.filters,
                                  async_job = args.async_job,
                                  output_file = f'{DATA_PATH}/table.vot.gz',
                                  output_format=args.output_format,
                                  verbose = args.verbose,
                                  get_query = args.get_query) 
def saveObservationIDs(observations):
    old_line = "yourMom"
    with open(f'{DATA_PATH}/observation_id.txt', 'w') as f:
        for line in observations:
            if "total" in line:
                pass
            # elif ends_with_pattern(line):
            #     pass
            elif old_line[:-3] in line or line[:-3] in old_line:
                pass
            else:
                f.write(f"{line}\n")
                old_line=line

def main():
    query_table=getHubbleQuery()
    saveObservationIDs(query_table['observation_id'])
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Query the Hubble Legacy Archive')
    parser.add_argument('--calibration_level', type=str, default='PRODUCT',
                        help='Calibration level of the data')
    parser.add_argument('--data_product_type', type=str, default='image',
                        help='Type of data product')
    parser.add_argument('--intent', type=str, default='SCIENCE',
                        help='Observation intent')
    parser.add_argument('--obs_collection', nargs='+', default=['HLA','HST'],
                        help='Observation collection')
    parser.add_argument('--instrument_name', nargs='+', default=['WFC3'],
                        help='Name of the instrument used')
    parser.add_argument('--filters', nargs='+', default=['F555W', 'F606W','F814W'],
                        help='List of filters used')
    parser.add_argument('--async_job', action='store_false',
                        help='Submit query as an asynchronous job')
    parser.add_argument('--query_id', type=str, default='tmp',
                        help='Name of output file')
    parser.add_argument('--output_format', type=str, default='votable',
                        help='Format of output file')
    parser.add_argument('--verbose', action='store_false',
                        help='Print verbose output')
    parser.add_argument('--get_query', action='store_true',
                        help='Print the query rather than submitting it')


    args = parser.parse_args()    
    hubbler = ESAHubble()
    DATA_PATH = os.path.join(QUERY_PATH,f"{args.query_id}")
    if not os.path.isdir(DATA_PATH): os.mkdir(DATA_PATH)
    main()