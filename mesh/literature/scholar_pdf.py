import requests
import subprocess
import tempfile
import difflib
import re
import unicodedata
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
import time
import random
import os
import json
import atexit

class ScholarSearch:
    USER_AGENT = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    cookie_file = "/tmp/scholar_cookies.txt"                                                                             

    GENERIC_COOKIES = (
        "ACM_Download=1; "
        "SpringerLink-Consent=accepted; "
        "WileyTermsAccepted=1; "
        "ElsevierWeb=accepted; "
        "NatureUser=anonymous; "
        "ROUTEID=.pdfdl"
    )

    def __init__(self, s2_api_key = None):
        self.cache = {}
        self.s2_cache = {}
        self.s2_api_key = s2_api_key
        
        self.load_cache()
        atexit.register(self.save_cache)
    # ---------------------------
    # Normalization helpers
    # ---------------------------

    def load_cache(self, path="scholar_cache.json"):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self, path="scholar_cache.json"):
        try:
            with open(path, "w") as f:
                json.dump(self.cache, f)
        except Exception:
            pass  # best effort

    def s2_request(self, url, params):                                                                                   
        """                                                                                                              
        Robust, cached, rate-limit-aware Semantic Scholar GET request.                                                   
        """                                                                                                              
        
        headers = {}                                                                                                     
        if self.s2_api_key:                                                                                              
            headers["x-api-key"] = self.s2_api_key                                                                       
         
        # generate cache key                                                                                             
        key = ("S2", url, tuple(sorted(params.items())))                                                                 
     
        # return cached                                                                                                  
        if key in self.s2_cache:                                                                                         
            return self.s2_cache[key]                                                                                    
     
        for attempt in range(5):                                                                                         
            try:                                                                                                         
                r = requests.get(url, params=params, headers=headers, timeout=8)                                         
             
                # soft rate limit -> retry with backoff                                                                  
                if r.status_code == 429:                                                                                 
                    time.sleep(1.0 * (attempt + 1))                                                                      
                    continue                                                                                             
             
                # crash on non-rate-limit errors                                                                         
                r.raise_for_status()                                                                                     
                data = r.json()
                # store in cache                                                                                         
                self.s2_cache[key] = data                                                                                
                return data                                                                                              
                                                                                                                      
            except Exception:                                                                                            
                time.sleep(1.0 * (attempt + 1))                                                                          
                                                                                                                      
        return None   
        
    def normalize_title(self, t):
        if not t:
            return ""
        t = ''.join(
            c for c in unicodedata.normalize('NFD', t)
            if unicodedata.category(c) != 'Mn'
        )
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip().lower()

    def extract_doi_from_url(self, url):
        if not url:
            return None
        match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", url)
        return match.group(0) if match else None

    # ---------------------------
    # PDF extraction
    # ---------------------------
    def extract_first_page_from_pdf(self, url, full_text=False):
        """
        Fetch PDF from URL and extract text.

        Args:
            url: URL to fetch PDF from
            full_text: If True, extract all pages; if False, just first page

        Returns:
            Extracted text or None if failed
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
            "Cookie": self.GENERIC_COOKIES,
            "Accept": "application/pdf,*/*"
        }

        try:
            r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            r.raise_for_status()
        except Exception:
            return None

        # Check if we got a PDF (either by content-type or magic bytes)
        content_type = r.headers.get("Content-Type", "").lower()
        is_pdf = "pdf" in content_type or r.content.startswith(b"%PDF-")

        if not is_pdf:
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(r.content)
                tmp.flush()
                tmp_path = tmp.name

            # Build pdftotext command
            cmd = ["pdftotext"]
            if not full_text:
                cmd.extend(["-f", "1", "-l", "1"])  # First page only
            cmd.extend([tmp_path, "-"])

            text = subprocess.check_output(cmd, text=True, timeout=30).strip()

            # Clean up temp file
            os.unlink(tmp_path)

            return text if text else None
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except:
                pass
            return None

    # ---------------------------
    # Publisher HTML extraction
    # ---------------------------
    def extract_publisher_abstract(self, url):
        """
        Fetch publisher HTML page and extract abstract using multiple strategies.

        Tries in order:
        1. JSON-LD structured data (Springer, Elsevier, Wiley)
        2. Publisher-specific selectors (ACM, etc.)
        3. Meta tags (citation_abstract, dc.Description)
        4. Generic text matching for "abstract" sections
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
            "Cookie": self.GENERIC_COOKIES,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }

        try:
            r = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            return None

        # Strategy 1: JSON-LD block (Springer, Elsevier, Wiley)
        ld = soup.find("script", type="application/ld+json")
        if ld and ld.string:
            try:
                data = json.loads(ld.string)
                # Handle both single object and array formats
                if isinstance(data, list):
                    data = data[0] if data else {}
                if isinstance(data, dict) and data.get("description"):
                    return data["description"].strip()
            except Exception:
                pass

        # Strategy 2: Publisher-specific selectors
        # ACM
        sec = soup.find("div", class_="abstractSection")
        if sec:
            return sec.get_text(" ", strip=True)

        sec2 = soup.find("section", class_="abstract")
        if sec2:
            return sec2.get_text(" ", strip=True)

        # Springer/Nature
        sec3 = soup.find("div", id="Abs1-content")
        if sec3:
            return sec3.get_text(" ", strip=True)

        # IEEE
        sec4 = soup.find("div", class_="abstract-text")
        if sec4:
            return sec4.get_text(" ", strip=True)

        # arXiv
        sec5 = soup.find("blockquote", class_="abstract")
        if sec5:
            return sec5.get_text(" ", strip=True).replace("Abstract:", "").strip()

        # Strategy 3: Meta tags used by various publishers
        meta_names = ["citation_abstract", "dc.Description", "DC.Description",
                      "description", "og:description"]
        for name in meta_names:
            el = soup.find("meta", {"name": name}) or soup.find("meta", {"property": name})
            if el and el.get("content"):
                content = el["content"].strip()
                # Only return if it looks like an abstract (not just a site description)
                if len(content) > 100:
                    return content

        # Strategy 4: Generic fallback - find paragraphs near "abstract" heading
        for heading in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            if "abstract" in heading.get_text().lower():
                # Look for next sibling paragraph
                next_p = heading.find_next(["p", "div"])
                if next_p:
                    text = next_p.get_text(" ", strip=True)
                    if len(text) > 50:
                        return text

        # Strategy 5: Look for any element with abstract in class/id
        for el in soup.find_all(["div", "section", "p"],
                                 attrs={"class": re.compile(r"abstract", re.I)}):
            text = el.get_text(" ", strip=True)
            if len(text) > 50:
                return text
        for el in soup.find_all(["div", "section", "p"],
                                 attrs={"id": re.compile(r"abstract", re.I)}):
            text = el.get_text(" ", strip=True)
            if len(text) > 50:
                return text

        return None

    # ---------------------------
    # Google Scholar scraping
    # ---------------------------

    def is_probable_pdf(self, url):
        if not url:
            return False

        p = urlparse(url)

        # Must have scheme + host
        if p.scheme not in ("http", "https"):
            return False
        if not p.netloc:
            return False

        # Direct PDFs
        if p.path.lower().endswith(".pdf"):
            return True

        # Reject Scholar internal nonsense
        if "scholar.google" in p.netloc:
            return False

        # URLs with pdf-ish paths from real domains
        if "pdf" in p.path.lower():
            return True

        return False


    def pick_best_pdf(self, urls):
        if not urls:
            return None

        # 1. direct .pdf links first
        exact = [u for u in urls if u.lower().endswith(".pdf")]
        if exact:
            return exact[0]

        # 2. arXiv / HAL
        for u in urls:
            host = urlparse(u).netloc
            if "arxiv.org" in host or "hal." in host:
                return u

        # 3. institutional (.edu, .ac.uk, .edu.cn, etc.)
        for u in urls:
            host = urlparse(u).netloc
            if host.endswith(".edu") or host.endswith(".ac.uk") or ".edu." in host:
                return u

        # 4. fallback — but still a real URL, not Scholar junk
        return urls[0]


    def scholar_search_raw(self, query, num_results=10):
        """
        Search Google Scholar and return raw results.

        Uses requests with session management and proper headers.
        Returns list of dicts with: title, authors_year, snippet, publisher_url, pdf_url
        """
        pages = list(range(0, num_results, 10))
        all_results = []

        # Use a session to maintain cookies across requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })

        for start in pages:
            time.sleep(random.uniform(0.8, 2.0))  # Longer delay to avoid detection
            url = f"https://scholar.google.com/scholar?q={quote_plus(query)}&hl=en&start={start}"

            if url in self.cache:
                html = self.cache[url]
            else:
                try:
                    r = session.get(url, timeout=20, allow_redirects=True)
                    r.raise_for_status()
                    html = r.text

                    # Check for CAPTCHA or block
                    if "gs_captcha" in html or "unusual traffic" in html.lower():
                        # Google Scholar is blocking us
                        break

                    # Only cache if we got valid results
                    if '<div id="gs_res_ccl"' in html:
                        self.cache[url] = html
                except Exception:
                    continue

            soup = BeautifulSoup(html, "html.parser")

            # Try multiple selector patterns (Scholar changes their markup)
            rows = (soup.select("div.gs_r.gs_or.gs_scl") or
                    soup.select("div.gs_ri") or
                    soup.select("div.gs_r"))

            for entry in rows:
                title_el = entry.select_one(".gs_rt")
                if not title_el:
                    continue

                title = title_el.get_text(" ", strip=True)
                # Remove [PDF], [HTML], [BOOK] prefixes
                title = re.sub(r"^\[(PDF|HTML|BOOK|CITATION)\]\s*", "", title, flags=re.I)

                author_year_el = entry.select_one(".gs_a")
                snippet_el = entry.select_one(".gs_rs")

                # Publisher link (external)
                pub_link = None
                for a in entry.select(".gs_rt a"):
                    href = a.get("href")
                    if href and "scholar.google" not in href:
                        pub_link = href
                        break

                # ---- PDF Collection ----
                pdf_links = []

                # PDF link from right-side "PDF" button
                side_pdf = entry.select_one(".gs_or_ggsm a")
                if side_pdf:
                    pdf_links.append(side_pdf.get("href", ""))

                # Backup: only strict .pdf endings from other <a>
                for a in entry.select("a"):
                    href = a.get("href", "")
                    if href.lower().endswith(".pdf"):
                        pdf_links.append(href)

                # Filter out garbage / Scholar nonsense
                pdf_links = [u for u in pdf_links if self.is_probable_pdf(u)]

                result = {
                    "title": title,
                    "authors_year": author_year_el.get_text(" ", strip=True) if author_year_el else "",
                    "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
                    "publisher_url": pub_link,
                    "pdf_url": self.pick_best_pdf(pdf_links),
                    "all_pdf_links": pdf_links,
                }

                all_results.append(result)

            if len(rows) < 10:
                break

        return all_results[:num_results]

    # ---------------------------
    # Semantic Scholar fallbacks
    # ---------------------------

    def semantic_scholar_by_doi(self, doi):
        if not doi:
            return None

        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": "title,abstract,year,authors,citationCount,openAccessPdf"}

        data = self.s2_request( url, params )

        if not data:
            return None
        if data.get("abstract"):
            return data["abstract"]

        return None

    def semantic_scholar_by_title(self, title):
        if not title:
            return None

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title,
            "fields": "title,abstract,year,authors,citationCount,openAccessPdf"
        }

        data = self.s2_request(url, params)
        if not data or "data" not in data:
            return None

        papers = data["data"]
        if not papers:
            return None

        target = self.normalize_title(title)

        # → Best case: exact match
        for p in papers:
            if self.normalize_title(p.get("title", "")) == target:
                abs_text = p.get("abstract")
                if abs_text:
                    return abs_text

        # → Fuzzy match fallback
        scored = [
            (difflib.SequenceMatcher(None, target,
                                     self.normalize_title(p.get("title", ""))).ratio(), p)
            for p in papers
        ]
        scored.sort(reverse=True)
        best_score, best_p = scored[0]

        if best_score > 0.75 and best_p.get("abstract"):
            return best_p["abstract"]

        # → Any abstract at all
        for p in papers:
            if p.get("abstract"):
                return p["abstract"]

        return None
    def semantic_scholar_fallback(self, title, publisher_url):
        # Step 1: try DOI, but do NOT do a title fallback afterwards
        doi = self.extract_doi_from_url(publisher_url)
        if doi:
            abs_text = self.semantic_scholar_by_doi(doi)
            if abs_text:
                return abs_text

        # Step 2: title match
        return self.semantic_scholar_by_title(title)

    # ---------------------------
    # Main search
    # ---------------------------

    def search(self, query, nresults=10 ):
        raw = self.scholar_search_raw(query, nresults)
        final = []

        for item in raw:
            abstract = None



            # 3. Semantic Scholar fallback
            if not abstract:
                abstract = self.semantic_scholar_fallback(
                    item["title"], item["publisher_url"]
                )

            # 1. PDF → first page
            #if item["pdf_url"]:
                #abstract = self.extract_first_page_from_pdf(item["pdf_url"])

            # 2. Publisher HTML
            if not abstract and item["publisher_url"]:
                abstract = self.extract_publisher_abstract(item["publisher_url"])
                abstract = abstract[0:1500]
                
            # 4. Scholar snippet
            if not abstract:
                abstract = item["snippet"]

            item["abstract"] = abstract
            final.append(item)

        return final
