import json
from zipfile import ZIP_DEFLATED, ZipFile


def json_dump_zip(filename: str, obj):
    with ZipFile(filename, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as zip_file:
        dumped_JSON: str = json.dumps(obj, ensure_ascii=False)
        zip_file.writestr("data.json", data=dumped_JSON)


def json_load_zip(filename: str):
    with ZipFile(filename, mode="r") as zip_file:
        with zip_file.open("data.json", "r") as fp:
            return json.load(fp)

__all__ = ["json_dump_zip", "json_load_zip"]


if __name__=="__main__":

    # Prepare some data
    data: dict = {
        "common_name": "Brongersma's short-tailed python",
        "scientific_name": "Python brongersmai",
        "length": 290
    }
    json_dump_zip("test.zip",data)

    data_2=json_load_zip("test.zip")

    pass

