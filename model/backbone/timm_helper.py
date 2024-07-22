from timm.models.registry import model_entrypoint

def is_model_registered(model_name):
    try:
      model_entrypoint(model_name)
      return True
    except:
      return False
