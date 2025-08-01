from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
import re



#------Step 1: Collect list of unique UK project IDS and associated sectors -----#
# Setup headless browser
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Remove this line if you want to see the browser
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)

# Load the UK-filtered project listing page
url = "https://golab.bsg.ox.ac.uk/knowledge-bank/indigo/impact-bond-dataset-v2/?query=&countries=United%20Kingdom"
driver.get(url)
time.sleep(5)

all_projects = [] 

def extract_projects_from_page():
    articles = driver.find_elements(By.XPATH, '//*[@id="indigo-react"]/div/div[2]/div/div[12]/div/article')
    for i, article in enumerate(articles, start=1):
        base = f'//*[@id="indigo-react"]/div/div[2]/div/div[12]/div/article[{i}]/div/div[1]'
        try:
            project_id = driver.find_element(By.XPATH, base + '/p[1]/span').text.strip()
            name = driver.find_element(By.XPATH, base + '/h3').text.strip()
            start_raw = driver.find_element(By.XPATH, base + '/p[2]').text.strip()
            stage_raw = driver.find_element(By.XPATH, base + '/p[3]').text.strip()
            sector_raw = driver.find_element(By.XPATH, base + '/p[4]').text.strip()

            project = {
                "Project ID": project_id,
                "Name": name,
                "Start Date": start_raw.replace("Start date:", "").strip(),
                "Stage": stage_raw.replace("Stage of development:", "").strip(),
                "Policy Sector": sector_raw.replace("Policy sector:", "").strip()
            }
            all_projects.append(project)
        except Exception as e:
            print(f"Error reading project {i}: {e}")

# Paginate through all results
page_number = 1
while True:
    print(f"Scraping page {page_number}...")
    extract_projects_from_page()
    page_number += 1
    try:
        next_page_xpath = f'//*[@id="indigo-react"]/div/div[2]/div/nav/ul/li[{page_number + 1}]/a'
        next_button = driver.find_element(By.XPATH, next_page_xpath)
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(5)
    except:
        print("No more pages/unable to find page", page_number)
        break

driver.quit()


df = pd.DataFrame(all_projects)
# Drop duplicates by Project ID
df = df.drop_duplicates(subset="Project ID")

print(f"Scraped {len(df)} unique projects")









#-----Step 2: Collect dates, capital, payments, and service users using list of unqiue IDs from above-----#


# Re-initialize headless browser before detail scraping
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)


# Add columns
df["Num Commissioners"] = 0
df["Num Investors"] = 0
df["Num Service Providers"] = 0
df["Num Intermediaries"] = 0

df["Contract Signed Date"] = ""
df["Service Start Date"] = ""
df["Completion Date"] = ""
df["Capital Raised"] = ""
df["Max Outcome Payment"] = ""
df["Service Users"] = ""

# --- Helper functions ---

def count_role_by_label(label_text):
    try:
        spans = driver.find_elements(By.TAG_NAME, "span")
        for span in spans:
            text = span.text.strip()
            if label_text.lower() in text.lower() and "(" in text:
                match = re.search(r"\((\d+)\)", text)
                if match:
                    return int(match.group(1))

        uls = driver.find_elements(By.TAG_NAME, "ul")
        for ul in uls:
            try:
                prev = ul.find_element(By.XPATH, 'preceding-sibling::*[1]')
                span = prev.find_element(By.TAG_NAME, 'span')
                if label_text.lower() in span.text.lower():
                    links = ul.find_elements(By.TAG_NAME, "a")
                    return len([a for a in links if a.text.strip()])
            except:
                continue
    except:
        pass
    return 0

def extract_by_label(label_text):
    try:
        paragraphs = driver.find_elements(By.XPATH, "//p[span]")
        for p in paragraphs:
            try:
                span = p.find_element(By.TAG_NAME, "span")
                if label_text.lower() in span.text.lower():
                    return p.text.replace(span.text, '').strip()
            except:
                continue
    except:
        pass
    return ""

# --- Main loop ---

for i, row in df.iterrows():
    pid = row["Project ID"]
    url = f"https://golab.bsg.ox.ac.uk/knowledge-bank/indigo/impact-bond-dataset-v2/{pid}/"
    print(f"Scraping {pid}...")
    driver.get(url)
    time.sleep(3)

    # Count stakeholders
    df.at[i, "Num Commissioners"] = count_role_by_label("Commissioners")
    df.at[i, "Num Investors"] = count_role_by_label("Investors")
    df.at[i, "Num Service Providers"] = count_role_by_label("Service Providers")
    df.at[i, "Num Intermediaries"] = count_role_by_label("Intermediary organisations")

    # Pull detail fields using <p><span>
    df.at[i, "Contract Signed Date"] = extract_by_label("Date outcomes contract signed")
    df.at[i, "Service Start Date"] = extract_by_label("Start date of service provision")
    df.at[i, "Completion Date"] = extract_by_label("Anticipated completion date")
    df.at[i, "Capital Raised"] = extract_by_label("Capital raised")
    df.at[i, "Max Outcome Payment"] = extract_by_label("Max potential outcome payment")
    df.at[i, "Service Users"] = extract_by_label("Service users")

# Save output
driver.quit()

#Convert capital raised and max outcome payment to numbers (GBP)
def convert_gbp_string_to_number(value):
    if pd.isna(value) or not isinstance(value, str):
        return ""

    # Remove all USD values
    value = re.sub(r'USD\s[\d.,]+[mk]?', '', value)

    # Find all GBP numbers in the string
    gbp_matches = re.findall(r'GBP\s([\d.,]+)([mk]?)', value)
    amounts = []
    for num, unit in gbp_matches:
        num = float(num.replace(",", ""))
        if unit == "m":
            num *= 1_000_000
        elif unit == "k":
            num *= 1_000
        amounts.append(num)

    # Return the max of all GBP values found (or empty string if none)
    if amounts:
        return int(max(amounts))
    return ""

df["Capital Raised"] = df["Capital Raised"].apply(convert_gbp_string_to_number)
df["Max Outcome Payment"] = df["Max Outcome Payment"].apply(convert_gbp_string_to_number)


#Clean service users column 
def clean_service_users(value):
    if pd.isna(value) or not isinstance(value, str):
        return ""

    # Remove the word 'individuals' and trim
    value = value.replace("individuals", "").strip()

    # Handle 'k+' format
    match = re.match(r"(\d+)k\+?", value)
    if match:
        return int(match.group(1)) * 1000

    # Handle plain integer strings like '10'
    match = re.match(r"(\d+)", value)
    if match:
        return int(match.group(1))

    return ""

df["Service Users"] = df["Service Users"].apply(clean_service_users)

print("Done with scrapping, info saved in df")


df.to_csv("uk_sib_projects_scraped.csv", index=False)
print("Saved to uk_sib_projects_scraped.csv")
