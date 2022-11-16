import json

from ptsemseg.loader.pascal_voc_loader                  import pascalVOCLoader
from ptsemseg.loader.camvid_loader                      import camvidLoader
from ptsemseg.loader.ade20k_loader                      import ADE20KLoader
# from ptsemseg.loader.mit_sceneparsing_benchmark_loader  import MITSceneParsingBenchmarkLoader
from ptsemseg.loader.cityscapes_loader                  import cityscapesLoader
from ptsemseg.loader.nyuv2_loader                       import NYUv2Loader
# from ptsemseg.loader.sunrgbd_loader                     import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader            import mapillaryVistasLoader
from ptsemseg.loader.railsem19_loader                   import RailSem19Loader                  # <added by jungwon>
from ptsemseg.loader.railsem19_LRC_loader               import RailSem19_LRC_Loader             # <added by jungwon>
from ptsemseg.loader.railsem19_C20_loader               import RailSem19_C20_Loader             # <added by jungwon>
from ptsemseg.loader.railsem19_seg_triplet_loader       import RailSem19_SegTriplet_Loader      # <added by jungwon>
from ptsemseg.loader.railsem19_seg_triplet_loader_b     import RailSem19_SegTriplet_b_Loader    # <added by jungwon>


########################################################################################################################
###
########################################################################################################################
def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        # "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        # "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "railsem19": RailSem19Loader,                                       # <added by Jungwon>
        "railsem19_LRC": RailSem19_LRC_Loader,                              # <added by Jungwon>
        "railsem19_C20": RailSem19_C20_Loader,                              # <added by Jungwon>
        "railsem19_seg_triplet": RailSem19_SegTriplet_Loader,               # <added by Jungwon>
        "railsem19_seg_triplet_b": RailSem19_SegTriplet_b_Loader            # <added by Jungwon>
    }[name]
#end
