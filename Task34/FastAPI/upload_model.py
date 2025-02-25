from huggingface_hub import upload_folder

upload_folder(
    folder_path="D:\\SourceCode\\ProjectOJT\\complete\\OJT_TASK3_LOCAL\\Deploy\\ban_va\\model_repository",  # Thư mục chứa model
    path_in_repo="",  # upload toàn bộ
    repo_id="hoanguyenthanh07/densenet_onnx",  
    repo_type="model",  
    token="hf_xFlrSSMBoHrYlSIvKwhKsSCakCPLknODsfssssahg"
)
