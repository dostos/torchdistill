import os
from pathlib import Path
import yaml

def get_student_model(models):
    student_str = Path(models['student']).read_text()

    student_str = student_str.replace('model_name', 'student_model_name')
    student_str = student_str.replace('model_experiment', 'student_model_experiment')

    student_str = student_str.replace('*model_experiment', "*model_experiment, '_from_', *teacher_model_name")

    return student_str

def get_teacher_model(models):
    teacher_str = Path(models['teacher']).read_text()

    teacher_str = teacher_str.replace('model_name', 'tracher_model_name')
    teacher_str = teacher_str.replace('model_experiment', 'tracher_model_experiment')

    return teacher_str

def get_plan(plan, alpha):
    plan_str = Path(plan).read_text()
    return plan_str.replace('&alpha', f'&alpha {alpha}')
    

def yaml_join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def yaml_pathjoin(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.expanduser(os.path.join(*[str(i) for i in seq]))


def load_yaml_file(yaml_file_path, custom_mode=True):
    if custom_mode:
        yaml.add_constructor('!join', yaml_join, Loader=yaml.FullLoader)
        yaml.add_constructor('!pathjoin', yaml_pathjoin, Loader=yaml.FullLoader)
    with open(yaml_file_path, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)

def load_yaml_files(dataset_path, models, plan, alpha, custom_mode=True):
    if custom_mode:
        yaml.add_constructor('!join', yaml_join, Loader=yaml.FullLoader)
        yaml.add_constructor('!pathjoin', yaml_pathjoin, Loader=yaml.FullLoader)

    yaml_str = Path(dataset_path).read_text() + '\n'
    yaml_str = yaml_str + get_plan(plan, alpha) + '\n'

    yaml_str = yaml_str + "models:\n"
    if isinstance(models, dict):
        yaml_str = yaml_str + "  teacher_model:\n" + get_teacher_model(models) + "\n"
        yaml_str = yaml_str + "  student_model:\n" + get_student_model(models) + "\n"
    else:
        yaml_str = yaml_str + "  model:\n" + Path(models).read_text() + "\n"

    return yaml.load(yaml_str, Loader=yaml.FullLoader)