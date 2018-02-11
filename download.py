################################################
# File name: download.py                       #
# Author: Mahmoud Badry                        #
# Date created: 2/11/2018                      #
# Date last modified: 2/11/2018                #
# Python Version: 3                            #
# Purpose: Download all notes in PDF format    #
# Requirements: pypandoc >= 1.4                #
################################################
import pypandoc


def main():
    marks_down_links = {
        "Standford CS231n 2017 Summary":
            "https://raw.githubusercontent.com/mbadry1/CS231n-2017-Summary/master/README.md",
    }

    # Extracting pandoc version
    print("pandoc_version:", pypandoc.get_pandoc_version())
    print("pandoc_path:", pypandoc.get_pandoc_path())
    print("\n")

    # Starting downloading and converting
    for key, value in marks_down_links.items():
        print("Converting", key)
        pypandoc.convert_file(
            value,
            'pdf',
            extra_args=['--latex-engine=xelatex', '-V', 'geometry:margin=1.5cm'],
            outputfile=(key + ".pdf")
        )
        print("Converting", key, "completed")


if __name__ == "__main__":
    main()
