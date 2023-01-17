
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
        print(filename, cleaned_files+"/cleaned_"+filename)
        with open(filename,'r') as infile, open(cleaned_files+"/"+filename,'w') as outfile :
            for line in infile :

                stripped_line_1 = regex.sub(" tuple\[[,a-zA-Z0-9\[\]\s]*]", " tuple",line)  # no type hinting for tuple
                stripped_line_2 = regex.sub("Callable\[[,a-zA-Z\[\]\s]*]","Callable",stripped_line_1)  # no type hinting for callable
                stripped_line_3 = regex.sub("Union\[[,a-zA-Z\[\]\s]*]", "Union",stripped_line_2)  # no type hinting for Union
                stripped_line_4 = regex.sub(" list\[[,a-zA-Z0-9\[\]\s]*]"," list",stripped_line_3)  # no type hinting for list
                stripped_line_5 = regex.sub(" dict\[[,a-zA-Z0-9\[\]\s]*,[,a-zA-Z0-9\[\]\s]*]"," dict",stripped_line_4)  # no type hinting for dict
                stripped_line_6 = regex.sub("->[.a-zA-Z\[\]\s]*:", " :", stripped_line_5)  # no type hinting for returns
                stripped_line_7 = regex.sub(":[.a-zA-Z\s]*,", ",", stripped_line_6)  # no type hinting for function args
                stripped_line_8 = regex.sub(":[.,a-zA-Z\s]*\)", ")", stripped_line_7)  # no type hinting for function args -- clash with array slicing
                stripped_line_9 = regex.sub(":[.,a-zA-Z\s]*=", "=", stripped_line_8)  # no type hinting for function args -- clash with array slicing
                stripped_line_99 = regex.sub("=\}", "}", stripped_line_9)  # python 3.6 has no {var=} in fstrings
                stripped_line = regex.sub("from __future__ import annotations", "", stripped_line_99)  # python 3.6 has no {var=} in fstrings
                outfile.write(stripped_line)


if __name__ == "__main__" :
    clean_src()
