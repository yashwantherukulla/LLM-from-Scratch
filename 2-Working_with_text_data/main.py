import re

def getText(filePath :str):
    try:
        with open(filePath, 'r', encoding="utf-8") as file:
            f = file.read()
            print(f"sucessfully got the file {filePath}\nThis file has {len(f)} characters")
            return f
    except:
        print("Error getting the file")

def tokenizeWithSpace(rawText: str) -> list:
    tokenizedText = re.split("\s", rawText)
    print(tokenizedText[:50])
    return tokenizedText

