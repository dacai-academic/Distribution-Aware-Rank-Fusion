import pandas as pd

from typing import Optional
from glob import glob


class ExcelFile(object):
    def __init__(self, path: str):
        self.path = path
        self.content = []
        self._load()

    def _load(self):
        excel_file = pd.ExcelFile(self.path)
        sheet_names = excel_file.sheet_names
        excel_file.close()
        for sheet in sheet_names:
            data = pd.read_excel(self.path, sheet_name=sheet)
            for i in range(1, data.shape[0]):
                question = str(data.iloc[i, 0])
                answer = str(data.iloc[i, 1])

                self.content.append([question, answer])
        return None


class Documents(object):
    def __init__(self, file_dir: Optional[str] = None, file_path: Optional[str] = None):
        assert (file_dir is not None) or (file_path is not None), \
            "please make sure that one of [$file_dir, $file_path] is provide!"
        self.content = []
        self.file_dir = file_dir
        self.file_path = file_path
        self._processing_excel()

    def _processing_excel(self):
        if self.file_path:
            assert "xls" in self.file_path.split(".")[-1], "please check the file format, only support [xls, xlsx]"
            ef = ExcelFile(self.file_path)
            self.content.extend(ef.content)

        else:
            xls_files = glob(self.file_dir + "*.xls")
            xlsx_files = glob(self.file_dir + "*.xlsx")
            for each in (xls_files + xlsx_files):
                ef = ExcelFile(each)
                self.content.extend(ef.content)
        return None


if __name__ == '__main__':
    # a = ExcelFile("/home/cai/project/rag_db/data/output/产品对比归类_2.0.xlsx")
    b = Documents(file_path="/home/cai/project/bm25_embedding/data/qa.xlsx")
    pass
