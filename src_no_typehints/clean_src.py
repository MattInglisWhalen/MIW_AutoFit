
import re as regex
import os as os

def clean_src():
    here =  os.path.abspath(__file__)
    cleaned_files = os.path.dirname(here)
    src_files = os.path.dirname( cleaned_files ) + "/src"
    os.chdir(src_files)
    print(src_files)
    for filename in os.listdir(src_files) :
        if filename[-3:] != ".py" :
            continue
        with open(filename,'r') as infile, open(cleaned_files+"cleaned_"+filename,'w') as outfile :
            for line in infile :
                stripped_line_1 = regex.sub("->[a-zA-Z\[\]\s]*:"," :",line)  # no type hinting for returns
                stripped_line_2 = regex.sub(":[a-zA-Z\[\]\s]*,", ",", line)  # no type hinting for function args
                stripped_line_3 = regex.sub(":[a-zA-Z\[\]\s]*\)", ")", line)  # no type hinting for function args
                outfile.write(stripped_line_3)


if __name__ == "__main__" :
    clean_src()
