from app.services.file_upload import file_upload_service


async def save_upload(file) -> str:
    file_path, _extension, _pipeline_type = await file_upload_service.validate_and_save(
        file, sub_dir="skeleton"
    )
    return file_path


def validate_image(file):
    return True

