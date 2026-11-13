from fastapi import APIRouter, Query

from app.core.errors import DatasetNotLoadedError
from app.ml.dataset_store import store
from app.ml.preprocessing import split_data

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def dataset_preview(n: int = Query(5, ge=1, le=100)):
    try:
        return {"n": n, "rows": store.preview(n)}
    except FileNotFoundError as e:
        raise DatasetNotLoadedError(str(e))


@router.get("/info")
def dataset_info():
    try:
        return store.info()
    except FileNotFoundError as e:
        raise DatasetNotLoadedError(str(e))
    except ValueError as e:
        raise DatasetNotLoadedError(str(e))


@router.get("/split-info")
def dataset_split_info():
    try:
        return split_data()
    except FileNotFoundError as e:
        raise DatasetNotLoadedError(str(e))
    except ValueError as e:
        raise DatasetNotLoadedError(str(e))