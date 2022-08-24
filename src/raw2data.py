import os
from os import path
from xml.etree import ElementTree as et

def xml_parser(file):
	content = ""
	dom = et.parse(file)
	root = dom.getroot()
	num = 0
	for sdrs in root.iter('sdrs'):
		num += 1
		for cond in sdrs.iter('con'):
			for c in cond:
				clause = "discourse " + str(num)
				cond_keys = c.attrib.keys()
				for k in cond_keys:
					if 'arg' in k or k == 'symbol':
						clause += " "
						clause += c.attrib[k]
				content += clause
				content += "\n"
	return content

#data
target_path_root =  r"../data/text_data"
n = 0
m = 0
data_path = r"../data/gmb-1.0.0/data"
for files in os.listdir(data_path):
	file_path = os.path.join(data_path, files)
	for folder in os.listdir(file_path):
		file_path2 = os.path.join(file_path, folder)
		for file in os.listdir(file_path2):
			file_path3 = os.path.join(file_path2, file)
			if file == "en.raw":
				content = open(file_path3, 'r', encoding='utf-8').read()
				text_path = "text" + str(n)
				text_path = os.path.join(target_path_root, text_path)
				text_f = open(text_path, 'a', encoding='utf-8')
				text_f.write(content)
				text_f.close()
				n += 1
			if file == "en.drs.xml":
				drs_content = xml_parser(file_path3)
				drs_path = "drs" + str(m)
				drs_path = os.path.join(target_path_root, drs_path)
				drs_file = open(drs_path, 'a', encoding='utf-8')
				drs_file.write(drs_content)
				drs_file.close()
				m += 1
