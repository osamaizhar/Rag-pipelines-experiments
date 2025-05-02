from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_400_BAD_REQUEST


# Custom handler for Pydantic/BaseModel validation
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Simplify to first error only for "single error style"
    first_error = exc.errors()[0]
    field = ".".join([str(loc) for loc in first_error["loc"] if loc != "body"])
    msg = f"{field}: {first_error['msg']}" if field else first_error["msg"]

    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"status_code": HTTP_400_BAD_REQUEST, "message": msg},
    )


# Custom handler for manually raised HTTPExceptions
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status_code": exc.status_code, "message": exc.detail},
    )
