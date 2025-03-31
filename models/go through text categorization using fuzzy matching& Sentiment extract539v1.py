import json
import os
import pandas as pd
import re
from datetime import datetime
import spacy
from rapidfuzz import process, fuzz
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
nltk.download('vader_lexicon')  # download the vader lexicon if the package is not already downloaded
sia = SentimentIntensityAnalyzer()

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# list of company to work on
cik_list = [1000180, 1000697, 1001039, 1001082, 1001250, 1001288,  1002047,  1002910,  1004155,  1004434,  1004440,  100493,  1004980,  100885,  1011006,  1011657,  1014473,  1015780,  101778,  101830,  1018724,  1018840,  1018963,  1020416,  1020569,  1021860,  1022079,  1024305,  1024478,  1026304,  1031296,  1032033,  103379,  1035002,  1035267,  1037038,  1037540,  1037949,  1038357,  1040971,  1041061,  104169,  1043277,  1043604,  10456,  1045810,  1047122,  1047699,  1047862,  1048286,  104889,  1048911,  1050915,  1053112,  1053507,  1054374,  1054833,  1056239,  1056288,  1058290,  1059556,  1060391,  106040,  1063761,  1064728,  1065088,  106535,  1065865,  1066107,  106640,  1070412,  1070750,  107263,  1073431,  107681,  10795,  1084580,  1084750,  1086222,  1087423,  108772,  1089976,  1090012,  1090727,  1090872,  109198,  109380,  1094093,  1099219,  1100962,  1101215,  1105705,  1108524,  1108827,  1109357,  1110783,  1110803,  1111711,  1111928,  1115222,  11199,  1120193,  1122304,  1124198,  1126294,  1126328,  1130310,  1133421,  1135152,  1135971,  1136893,  1137411,  1137774,  1141391,  1141982,  1144215,  1156375,  1158449,  1163165,  1163739,  1164727,  1166126,  1166691,  1168054,  1169055,  1170650,  1174746,  1174922,  12659,  1267238,  1274057,  1274494,  1275283,  1276520,  1279363,  1281761,  1288776,  12927,  1310067,  1324404,  1324424,  1326160,  1326380,  1336917,  1341439,  1358071,  1359841,  1361658,  1365135,  1368007,  1373835,  1377013,  1378946,  1385187,  1390777,  1393311,  1393612,  1393744,  1396009,  1399315,  1403161,  1408146,  1413329,  1418091,  1418135,  1424847,  14272,  1430602,  1451505,  1452575,  1457543,  1458891,  1465112,  14693,  1478242,  1490892,  1491675,  1492633,  1495569,  1496048,  1518832,  1519751,  1520566,  1526520,  1532063,  1545158,  1546640,  1551182,  1567892,  1571949,  1574540,  1575571,  1575828,  1578318,  1578732,  1585364,  16732,  16918,  1800,  18230,  19617,  200406,  201533,  20286,  203077,  203527,  20520,  21076,  21344,  216228,  21665,  217346,  23082,  23217,  24545,  24741,  2488,  26172,  27419,  277135,  277948,  28412,  29669,  2969,  29905,  29915,  29989,  30371,  30554,  30625,  310158,  310764,  31277,  313616,  31462,  314808,  315189,  315213,  315293,  315852,  316206,  316709,  317187,  317771,  318154,  319201,  320187,  320193,  32604,  33185,  33213,  34088,  34408,  350563,  350698,  352049,  353944,  354908,  354950,  35527,  356028,  36104,  36270,  3673,  36966,  37748,  37785,  37996,  38074,  38777,  39911,  40533,  40545,  40704,  40987,  42542,  42582,  4281,  4447,  45012,  46080,  46640,  46765,  47111,  47217,  48039,  48465,  4904,  49196,  4962,  4977,  49826,  50863,  51143,  51253,  51434,  51644,  5272,  53456,  54480,  55067,  5513,  55785,  56873,  58492,  59478,  59558,  60086,  60667,  62709,  6281,  62996,  63276,  63754,  63908,  64670,  64803,  65350,  65984,  66479,  66740,  67472,  6769,  68505,  6951,  701221,  702165,  70318,  703360,  704051,  70530,  7084,  70858,  712515,  713676,  71691,  717423,  718877,  719739,  72020,  721083,  721371,  721683,  72207,  723125,  723254,  72333,  723531,  72903,  72971,  730464,  73124,  731766,  732485,  732712,  732717,  73309,  7332,  740260,  743316,  743988,  745732,  746515,  750556,  751652,  753308,  75362,  754737,  75488,  758004,  75829,  76334,  764065,  764180,  764478,  764622,  766421,  768251,  768835,  769397,  773840,  773910,  77476,  78003,  78239,  783280,  783325,  78814,  788784,  789019,  789073,  790070,  791519,  791907,  793952,  794323,  796343,  79732,  797468,  79879,  79958,  800240,  800459,  801898,  804055,  804212,  80424,  804328,  804753,  80661,  811156,  812074,  814453,  815094,  815097,  815556,  816284,  816761,  818479,  820027,  820096,  820313,  821189,  822416,  823768,  826083,  827052,  827054,  829224,  831001,  832988,  833444,  835729,  836106,  836267,  849213,  849636,  850209,  850693,  85408,  855683,  858877,  859014,  85961,  86144,  861878,  86312,  863157,  864328,  865436,  866787,  8670,  868809,  873364,  87347,  874761,  874766,  875045,  875159,  877890,  879101,  8818,  882095,  882184,  882835,  883569,  885639,  885721,  885725,  886158,  8868,  886982,  890801,  891024,  895126,  895421,  895648,  896159,  896878,  89800,  898173,  898293,  899051,  899689,  899866,  899881,  90185,  906107,  906709,  909832,  909954,  912093,  912242,  912615,  912750,  91419,  914208,  91440,  915389,  91576,  915912,  916863,  918160,  91882,  920148,  920760,  920990,  92122,  921738,  921847,  922224,  922864,  92380,  927066,  927628,  927653,  929887,  931336,  933136,  93410,  93556,  936340,  936468,  93751,  940944,  941548,  945436,  945764,  949039,  949189,  95304,  95521,  96021,  96289,  97210,  97476,  97745,  98246,  9892]

#List = pd.DataFrame(List)
#List['list'] = List['list'].astype(str)
#a list of the company we can to run



#Name_df = pd.DataFrame(Name_df)
file_path ="/home/fab523/pjz218_lorallm_proj/shared/Fuzzy Matching/CIKfile.csv"
Name_df= pd.read_csv(file_path)
# Ensure CIK column is string
Name_df['CIK'] = Name_df['CIK'].astype(str)

# Define categories and their keywords
categories = {
    "BUSINESS COMBINATION AND RESTRUCTURING": ["merger", "acquisition", "joint venture", "separation", "spin-off"],
    "FINANCIAL ACTIVITIES": ["lend", "borrow", "loan", "notes", "payment", "debt", "stock", "dividend", "asset-backed securities (ABS)"],
    "OPERATION ACTIVITIES": ["operation", "contract", "consulting", "service", "product", "supply"],
    "SENIOR PERSONNEL CHANGE": ["executive officer/director", "retire", "leave", "appointment"],
    "INFORMATION DISCLOSURE": ["conference", "presentation", "statement", "exhibit"],
    "DOCUMENT UPDATE": ["by-laws", "code of ethics"],
    "INTELLECTUAL PROPERTY ACTIVITIES": ["intellectual property", "patent approval"],
    "LITIGATION AND LAWSUIT": ["settlement", "litigation", "lawsuit"],
    "DELISTING, TRADING SUSPENSION": ["delisting", "trading suspension"],
    "BANKRUPTCY": ["bankruptcy"],
    "NONE": ["No material event happened"],
    "SHARE BUYBACK/REPURCHASE PROGRAM BEGIN": ["initiates share repurchase", "announces stock buyback program", "authorizes equity repurchase", "board approval for stock repurchase", "launches share repurchase", "commences share buyback"],
    "SHARE BUYBACK/REPURCHASE PROGRAM UPDATE": ["increases buyback authorization", "expands repurchase program", "extends buyback period", "updates on share repurchase", "additional share repurchase", "accelerates stock buyback"],
    "SHARE BUYBACK/REPURCHASE PROGRAM SUSPENSION": ["suspends share repurchase", "halts stock buyback", "pauses equity repurchase", "temporary suspension of buyback", "delays stock repurchase program", "moratorium on share repurchases"],
    "SHARE BUYBACK/REPURCHASE PROGRAM CONCLUSION": ["concludes share repurchase", "completes stock buyback", "finalizes equity repurchase", "ends stock repurchase program", "termination of buyback program", "wrap-up of share repurchases"]
}

# Function to preprocess text
def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# Function to categorize a document with a focus on specific repurchase announcements and programs
def categorize_document(doc):
    category_scores = []
    for category, keywords in categories.items():
        scores = process.extract(doc, keywords, scorer=fuzz.partial_ratio)
        if scores:
            max_score = max([score[1] for score in scores])
            if category.startswith("SHARE"):
                max_score *= 1.2  # Increase the weight for share repurchase category
            if max_score < 90:
                category = "NONE"
            category_scores.append((category, max_score))
            
    
    # Sort by score in descending order and return top 3 categories
    category_scores.sort(key=lambda x: x[1], reverse=True)
    top_categories = category_scores[:3]
    return top_categories
   

def extract_date_from_filename(filename):
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # Matches YYYY-MM-DD
        r'(\d{4}-\d{2})',        # Matches YYYY-MM
        r'(\d{8})',              # Matches YYYYMMDD
        r'(\d{4}\.\d{2}\.\d{2})' # Matches YYYY.MM.DD
    ]

    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            date_str = match.group(1)
            try:
                if '-' in date_str:
                    return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                elif '.' in date_str:
                    return datetime.strptime(date_str, '%Y.%m.%d').strftime('%Y-%m-%d')
                elif len(date_str) == 8:  # For YYYYMMDD
                    return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                else:  # For YYYY-MM
                    return datetime.strptime(date_str, '%Y-%m').strftime('%Y-%m')
            except ValueError:
                continue
    return None


def cleaning(raw_date):
    cleaned= " ".join(raw_date.split())
    return cleaned 
def format_date(raw_date):
    cleaned_data= cleaning(raw_date)
    format_list = ['%B %d, %Y', '%d %B %Y']
    for dformat in format_list:
        Good = datetime.strptime(cleaned_data, dformat)
        return Good
    

def extract_data_from_report(report_text, filename):
    company_name = "" # Placeholder for company name extraction logic
    report_date = ""
    keywords_found = []
    item_numbers = []

    # Extract report date from content
    date_patterns = [
        r'Date of Report \(Date of Earliest Event Reported\): (\w+ \d+, \d{4})',
        r'Date of Report \(Date of earliest event reported\): (\w+ \d+, \d{4})',
        r'Date of Report \(Date of earliest event reported\): (_?\w+ \d+, \d{4})',
        r'Date of Report \(Date of Earliest Event Reported\):\s+(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of Report \(Date of earliest event reported\):?\s+(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of report \(date of earliest event reported\):?\s+(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of report \(Date of earliest event reported\):?\s+(\w+\s+\d{1,2},\s+\d{4})',
        r'\*{2}Date of report \(Date of earliest event reported\):\s*(\w+\s+\d{1,2},\s+\d{4})\s*\*{2}',
        r'\*{2}Date of Report \(Date of earliest event reported\):\s*(\w+\s+\d{1,2},\s+\d{4})\s*\*{2}',
        r'Date of Report \(Date of earliest event reported\):\s*\n*\n*(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of Report \(Date of earliest event reported\):\s*(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of Report \(Date of earliest event reported\):\s*\*{2}(\w+\s+\d{1,2},\s+\d{4})\*{2}',
        r'Date of Report \(Date of Earliest Event Reported\):\s*\|\s*\|\s*(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of Report \(Date of earliest event reported\)\s*\*{2}(\w+\s+\d{1,2},\s+\d{4})\*{2}',
        r'Date of R[eE]port \(Date of Earliest Event R[eE]ported\):\s+(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of Report \(Date of Earliest Event Reported\): (\w+ \d{1,2}, \d{4})',
        r'Date of Report \(Date of earliest event reported\):  (\w+ \d{1,2}, \d{4})',
        r'Date of Report \(Date of earliest event reported\) (\w+ \d{1,2}, \d{4})',
        r'Date of Report \(Date of earliest event reported\)\s*(\w+\s+\d{1,2},\s+\d{4})\s*\*{2}',
        r'DATE OF REPORT \(DATE OF EARLIEST EVENT REPORTED\):\s*(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of Report:\*{2}\s+(\w+\s+\d{1,2},\s+\d{4})\*{2}',
        r'Date of Report:\*{2}(\w+\s+\d{1,2},\s+\d{4})\*{2}',
        r'Date of Report:\s*\*{2}(\w+\s+\d{1,2},\s+\d{4})\*{2}',
        r'Date of Report:\s*\*{2}(\d{1,2}\s+\w+\s+\d{4})\*{2}',
        r'Date of Report:\s+(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of report:\s+(\w+\s+\d{1,2},\s+\d{4})',
        r'Date of Report:\s*\*{2}(\w+\s+\d{1,2},\s+\d{4})\*{2}',
        r'Date of Report:\s*\*{2}(\d{1,2}\s+\w+\s+\d{4})\*{2}',
        r'Date of Report.*?: (\w+ \d+, \d{4})',
        r'Date of Report \(Date of earliest event reported\): (\w+ \d+, \d{4})',
        r'\*{2}(\w+\s+\d{1,2},\s+\d{4})\*{2}\s*\n*\n*\*{2}Date of Report \(Date of earliest event reported\)\*{2}',
        r'(\w+ \d+, \d{4})\s+Date of Report \(Date of earliest event reported\)'
        #r'Dated: (\w+ \d{1,2}, \d{4})',
        #r'Date: (\w+ \d{1,2}, \d{4})',
]
    for pattern in date_patterns:
        match = re.search(pattern, report_text)
        if match:
            raw_date = match.group(1)
            try:
                #report_date = datetime.strptime(raw_date, '%B %d, %Y').strftime('%m/%d/%Y')
                report_date = format_date(raw_date.lstrip('_')).strftime('%Y-%m-%d')
            except ValueError as e:
                print(f"Error parsing date: {e}")
            break

    # Fallback to filename if date not found in content
    if not report_date:
        report_date = extract_date_from_filename(filename)

    # Extract item numbers
    item_pattern = r'Item\s(\d+\.\d{2})'
    found_items = re.findall(item_pattern, report_text)
    item_numbers.extend(found_items)

    # Preprocess text and categorize document
    preprocessed_text = preprocess(report_text)
    top_categories = categorize_document(preprocessed_text)

    # Extract keywords found
    for keywords in categories.values():
        for keyword in keywords:
            if keyword in report_text:
                keywords_found.append(keyword)

    # Perform sentiment analysis on the entire text
    sentiment_score = sia.polarity_scores(report_text)
    
        
        
    return company_name, report_date, top_categories, keywords_found, sentiment_score, item_numbers

def extract_folder_name(folder_path):
    return os.path.basename(folder_path)

def extract_cik(dirpath):
    N = os.path.dirname(dirpath)
    return os.path.basename(N)

def process_reports(folder_path):
    data = []
    print(f"cik_list: {cik_list}")
    
    for main_subdir in os.listdir(folder_path):
        main_subdir_path = os.path.join(folder_path, main_subdir)
        if not os.path.isdir(main_subdir_path):
            continue
        
        for sub_subdir in os.listdir(main_subdir_path):
            sub_subdir_path = os.path.join(main_subdir_path, sub_subdir)
            if not os.path.isdir(sub_subdir_path):
                continue

            for filename in os.listdir(sub_subdir_path):
                full_path = os.path.join(sub_subdir_path, filename)
                folder_name = extract_folder_name(sub_subdir_path)
                vr = extract_cik(sub_subdir_path)

                try:
                    if int(vr) in cik_list:
                        if filename.endswith('.txt') and '(8-k)' in filename.lower():
                            with open(full_path, 'r', encoding='utf-8') as file:
                                report = file.read()
                                # Concatenate all pages of the report text
                                report_text = " ".join(report.splitlines())
                                company_name, report_date, top_categories, keywords_found, sentiment_score, item_numbers = extract_data_from_report(report_text, filename)
                                CIK = extract_cik(sub_subdir_path)
                                ID= folder_name
                                data.append({
                                    'CIK': CIK,
                                    'ID': ID,
                                    'Report Date': report_date,
                                    'cat1': top_categories[0][0],
                                    'cat2': top_categories[1][0],
                                    'cat3': top_categories[2][0],
                                    #'Top Categories': ", ".join([f"{cat[0]} ({cat[1]})" for cat in top_categories]),  # Top categories with scores
                                    'Keywords Found': ", ".join(set(keywords_found)),  # Remove duplicates
                                    'Sentiment': sentiment_score['compound'],  # Compound score combines all scores
                                    'Item Numbers': ", ".join(set(item_numbers))  # Remove duplicates
                                })
                except ValueError:
                    print(f"Skipping directory: {vr}, not a valid CIK")

    df = pd.DataFrame(data)
    #print(f"df:\n{df}")
    # Ensure CIK column is string
    df['CIK'] = df['CIK'].astype(str)
    return df

#def lookup(df, Name_df, lookup_col, return_col):
    #return df.merge(Name_df[[lookup_col, return_col]], on=lookup_col, how='left')

# Path to the folder containing the reports
folder_path = "/home/fab523/pjz218_lorallm_proj/shared/sec_gov/Archives/edgar/textTest9"
df = process_reports(folder_path)
#df2 = df[df['CIK'].isin(List['list'])]

#result = lookup(df, Name_df, 'CIK', 'Company')
output = pd.DataFrame(df)
output['Sentiment'] = output['Sentiment'].astype(float)
output['Sentiment'] = ["positive" if x > 0.5 else "negative" if x < -0.5 else "neutral" for x in output['Sentiment']]
output['filling date from foldername']= output['ID'].str.replace('_'," ")
output[['Filling date F', 'Published date F', 'Counter']] = output['filling date from foldername'].str.split(expand=True)
output = output.drop('filling date from foldername', axis=1)
output.to_excel('FuzzyM539Companies.xlsx', index=False)

