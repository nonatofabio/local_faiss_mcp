import pytest
import urllib.request


@pytest.fixture(scope="session")
def test_pdfs(tmp_path_factory):
    """Download test PDFs for metadata extraction testing."""
    pdf_dir = tmp_path_factory.mktemp("test_pdfs")

    pdfs = {
        "dummy.pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "test.pdf": "https://www.orimi.com/pdf-test.pdf",
    }

    downloaded = {}
    for name, url in pdfs.items():
        path = pdf_dir / name
        try:
            urllib.request.urlretrieve(url, path)
            # Validate it's actually a PDF
            with open(path, "rb") as f:
                if f.read(5) != b"%PDF-":
                    continue
            downloaded[name] = path
        except Exception:
            pass

    if not downloaded:
        pytest.skip("Could not download any test PDFs")

    return downloaded
