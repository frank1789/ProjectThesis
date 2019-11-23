#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import pandas as pd


def xml_to_csv(path):
    xml_list = []
    xml_df = None
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            filename = root.find("filename").text
            width = root.find("size")[0].text
            height = root.find("size")[1].text
            class_name = member.find("name").text
            xmin = member.find("bndbox/xmin").text
            ymin = member.find("bndbox/ymin").text
            xmax = member.find("bndbox/xmax").text
            ymax = member.find("bndbox/ymax").text
            xml_list.append((filename, width, height, class_name, xmin, ymin, xmax, ymax))
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="xml_to_csv utility to convert from annotations.xml to annotation.csv")
    parser.add_argument('-a', '--annotations', required=True,
                        metavar="/path/to/dataset/",
                        help="Path to dataset folder")
    args = parser.parse_args()
    image_path = os.path.join(args.annotations, 'annotations')
    if not os.path.exists('raccoon_labels.csv'):
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('raccoon_labels.csv', index=None)
        print('Successfully converted xml to csv.')
    else:
        pass


if __name__ == "__main__":
    main()
    sys.exit()
