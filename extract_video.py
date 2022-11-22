import models.video_extractor
from optparse import OptionParser
from tools.config_tools import Config

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--config',
                      type=str,
                      help="video extract configuration",
                      default="./configs/video_extractor.yaml")

    (opts, args) = parser.parse_args()
    assert isinstance(opts, object)
    opt = Config(opts.config)
    print(opt)

    if hasattr(models, opt.model):
        model_class = getattr(models, opt.model)
        model = model_class(opt)
    else:
        raise ModuleNotFoundError(f"No implementation of {opt.model}")

    model.extract_dir(opt.dirname, opt.cuda)