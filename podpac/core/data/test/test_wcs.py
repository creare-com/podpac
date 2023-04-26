import pytest
import traitlets as tl
from io import BytesIO
import numpy as np

import podpac
from podpac.core.data.ogc import WCS

COORDS = podpac.Coordinates(
    [podpac.clinspace(-132.9023, -53.6051, 100, name="lon"), podpac.clinspace(23.6293, 53.7588, 100, name="lat")],
    crs="+proj=longlat +datum=WGS84 +no_defs +vunits=m",
)


class MockClient(object):
    """Mocked WCS client that handles three getCoverage cases that are needed by the tests below."""

    def getCoverage(self, **kwargs):
        if kwargs["width"] == 100 and kwargs["height"] == 100:
            return BytesIO(
                b"II*\x00\x08\x00\x00\x00\x14\x00\x00\x01\x03\x00\x01\x00\x00\x00d\x00\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00d\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\xfe\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\x06\x01\x00\x00\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00(\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00=\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00B\x01\x03\x00\x01\x00\x00\x00\x00\x01\x00\x00C\x01\x03\x00\x01\x00\x00\x00\x00\x01\x00\x00D\x01\x04\x00\x01\x00\x00\x00\xe6\x01\x00\x00E\x01\x04\x00\x01\x00\x00\x00\xad\n\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\xd8\x85\x0c\x00\x10\x00\x00\x00\x0e\x01\x00\x00\xaf\x87\x03\x00 \x00\x00\x00\x8e\x01\x00\x00\xb0\x87\x0c\x00\x02\x00\x00\x00\xce\x01\x00\x00\xb1\x87\x02\x00\x08\x00\x00\x00\xde\x01\x00\x00\x00\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00F\x029\x9f\xa4\xa1\xe9?\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00J\x82\x8fv\xb0\xa9`\xc0\x00\x00\x00\x00\x00\x00\x00\x00\xf26`\xb3Gz\xd3?\x00\x00\x00\x00\x00\x00\x00\x00\x03\x9f\xa0>%z7@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x01\x00\x01\x00\x00\x00\x07\x00\x00\x04\x00\x00\x01\x00\x02\x00\x01\x04\x00\x00\x01\x00\x01\x00\x00\x08\x00\x00\x01\x00\xe6\x10\x01\x08\xb1\x87\x07\x00\x00\x00\x06\x08\x00\x00\x01\x00\x8e#\t\x08\xb0\x87\x01\x00\x01\x00\x0b\x08\xb0\x87\x01\x00\x00\x00\x88mt\x96\x1d\xa4r@\x00\x00\x00@\xa6TXAWGS 84|\x00x^\xed\x9bypU\xd5\x1d\xc7\xcfK$+/\x84$$B00\x01\x92\x82\x18\x13\x08\x892M\x14B5\xc4@\x1d\x86\x01\xa4\x80\xd0\x8cc\x19\x1b\xa0X\x96\nXp\xa6S\x8dC\x85\xa1VF@\xa9X+\xadE\x8b\x10Y\x1ai\xd3\xa4\x02\x06\xd4D\x89\xefE I\x13\xb2\x02%{x\xf0n\xcfr\x97s\x97\x87\xa9\xd3?r\xcf\xfb\x9d?\xf2\xee;K\xe6~?\xbf\xe5,\xf7>\x84|\x17\xc9\x81\xdb$\xae\x9d|\xf7\xa3\xe2\xe7\xfa\x07.\xbf5VD\xaf\xf0s\xfd\x122\x86?\xad\xf0\x9bB\xf5\xf3\xd9\x0fY\xea\xef\r\x15\x95\x08\xf6\xffo\x97\x8fh\x94\x08YL\xfa\xad\xec\xcf\xa2DD\xfd\xc4\xb0z\xfb[\xeb\xb7\xaa\x15\x81\xc7\xc0\xf4cDb\x06\x80E\xfa\xb3TJ\xa2D@\x02\xd8\xf3\x8d\xe9\xcf2\xff\xd3~\xe2\x15\x12\xf8\x03\xd4/\xa2|\x9c\xf9\x88,C\xfa\xb3\xb0\xb4\xe4\x10\xd1\xf9\x89t\xd0o\xb4>\r\tC\x11\xd9\xfe\x03\xd1/;\x8a\x11\x8b\xfd\xbfc\xc3Z\x888\xd0\xfb\xb4\xbeV\xd0\xf4OV5F\xfd}!\x08]iM#\xfa\xbb\xc3\x15\n\x82\xae~\x88,\xa3~\x84\\\xc95Ig3\xf2\x8a\xe7\\\xfe\xd2\x9f\xf5wF\x8a\xaf\xdf\xbc\xf7\xe9:\xb2\xe8\xcd\x158*>\xfbW\xcd\x91\x9e&j\x7f\x92#L3\x82>?\xd8\xf4\x9bi\xef\xe3\xfa2`z\x9c\x9c\x12\x1e\xbf0\xebw\xea\xea@H\xfd\xa6\xbd\x8f\xe48\xbe\x7fM\xe6\x9e\xa7hV\\\xb6q\x92bVAW\x7f\xe6\xf4\xd7[5a\xb82%\x1c\\\xc8y\xb5\xa8\xf67g\x7f\xd4\xb5\xcf\xf9c\xbc\x04\xc4-\xaf\xad\xd4\x08\x88\x08\xc0b\xf6\xc7\x82\xfdI\xbfe\xde\xbe1\x8cV\xfflG\x87Sh\xfb[\xf8>\xd3\xbb\xe1%\xf2Y\xf4'\xc7\xb6<m}(^\x00X\xeb\xbf\x1aM!\xe0\xec\xf7PX\xfc^\x92\x04\xe9|X\xf8[\xcd\x17D\xb9\xb2\x04P\x9f@\x05\xff\xe3!\xf7\x85s\x01\x0f\xa3\x19T,\xa9\xf1\x13\x07\xe0\xf4\xefI\x17Z?3\xff\x95Q&w>\xfe(\xa9j}\xad2c\x03kc+B\xc1\x1c\xc0z\xf6ChW!\xfaO\xe4\x8d\xbeXG\x91t\xf5\xa7\t\x1c\x1c\xc1\xf4[\xec|\xa9\xd8\xd2l\xa6\xf9\xeb\x1a\xf7\x17\x8b\xf2Du\x00\xeb\x83\x1f\xac\xf6\xf3\xb0$\xa6\xf9\x99+?\xfcj\xea\"]x\x88\xe3\x01r\xeaO\xa8\xe7\xfc\x9b\xbf\xbc\x1d\x88>\xae\xa8\xed)\x8a\xd9]\xf7t\xa2\xbc!\xc4\xed\xc2\x00\xf8_\xf4w\xa6\xaa\x13\xa5h\xfa5\x93S\x1b\xd7\xba\x1fQk>\xef\x0f*n\n\xbf\x9e?\x17y\x03\xd4JQ\xf4\xfb\xca\xfd\x8a\xd0\x86\xd1\xf8\xeaX\xcb\xce\xe0'C\xbe\x97\x1a\x82\x83A)\x82\x00\xa8\xad\x99\xa5\x99\xde\xe2\xea\xeb\xaet\x84\xce\xecN>z\xf2\x0fI=\x8f\xa2\xfe`\xb9\x8f \xf2\xa5\x0f\xe7X\x88&U\xbf\x7f\x12/\xfd'\xdd\xe7xn\xad\xfbA\xfcu\xda\xe6\x0b\xcf\x95Ow4\xc7\xb1\xee\xc2\x9c\x03\xa9\t\xcd\n\xc3+\xcb\x87\xbf\xd1xq?B\xef,\x9e?\xae\xa9o\x07\xb7@\x14\xc4\xfe\xbe\x96>\x8c\x86\xaa\xdf\x9d4\x7f\xdc%\x87\x80\xfau\x8b\x1fuvW\x16~E\xeb\xd1\xb6\x86\xec\xe0\x05\x08m?6:>4+\xe8\x01\xd5K\x04\xb1\xbft\xeb.+\xc7G\xf8\xe4\x97\x14\xcfo*vtV\xdf\x93\xde\xf4\xe0\xe4\xa0a\xe1\xd3+_\xc4uG\x83~\x80\xff\n\xa2\xff\xc3|\xed\\G\x07\xa2\xd3\xf9\xf2:\x9c\x00?\xea.\xc8m\x1ay\xee\xc4!\xe7\x9a\xbd\x19\x11\xfd\x91\x18\xcb\x99L\xccL\x10\xf9R\xf5DK\xf3\xa3\x9a\xe2\xd5+&\xad[\xd0>\xf3\xaf\x93\x12\xb7\xee[\xb1\xaalJv[G\xdd[O\xbd\xaet\x17\x03\x80$\x9fq\x99!\xb8\xab\\\x93\xe7\xbeZ\\\x7f3z\xd8\xbc\x8b\xbd\xae\x0c)\xbc>\xba>,*l\xf3\x8d\x08\xda\xd9\x13d\x1ec\xbf\x9a\x01\xe8\x1f\xe1\xb4\xd6?D\x00\x0f\x90\xd0\xa9\x19>\x8cV\xf2E\xdf\xd0\xfc_\x8e+\x99\xb5u6\x9aR\xbbvW\xc6\x98\xb0\xb2NOH6\x1a?\xbe\xa74\x97\x9c\x03\n\xa1\x7f_\x81\x0f\xfd'\x1b\x82\xd2V?\xfbq\xc4\xe6y\xb1\r\t\x15\x8d\x81S\x93_,\xfc\xc5\x07\xcf\xbc\xe7\xbc\xd2\xf7\x13\x84*\xa6\t\xf1\x0e\xa4\x84\xceO\xe1_\xee\x90Y\x90u@\xedXt\xf8\x8dxWr\xab\xb3z\xc5\x9f\xa3\xbb\xcf\xe4\xe4\x04_\xcc\x8c\xbf\xffLO\xfbc\xaeT\x84\xae\xb6'\x0b0\x07\xd0\xc5\xafv\xa6\xa1w\x85\xc6\xca\x03\x89C\x8b\xefo\xcd_:g\xd3\x03i\xc1\x0b\x9bz\x97\xdfN)\xfd\xea\xf2\xae\x9d\xab\xd80\xfb\x07\x80\x9f\xeb\xbf\xe3\xde\x07\xb5\xfd{\xfb\x84\xde\xfeOF\xaf\x9c\xd5\xfc#\xef\xa9\x99\xa3Rn\xa6uD\xb4\xf4y\xafv\x17$\xe3\xeda\xfa\xbd\x82\xd8_\xf5\xfa\xc6x\xf5\x92\xc4DWk\xf8\x86\xd0\xf2\x88\xd8\xa0\xcb)\xb7\xfe\xf6^Y\xf9!\xf4\xbasb]\xc7R\xb7+3\x16\x95\xe4\x90\xc8\xd1\x07\x8c\xfd\xbe\xe9\x1d\xa0\xf9n}.\xb8\x16U\xb5\xf7\xe7\xcb\"k\xbf\x7fdd\xdf\xd8\xa0w\xb1\xbc\x0fn%wx\xb3\x96\xbd0\xf6Z\x94\x10)@\xa7\xdf*\x0f\xae\xaek\xba\xfb^o\xeb\x85\x90\xbf/^\x8b\xcf\x81\x10zy=\xaai\x17f\x17\xa8\xb7?\x99\x0b\r%\xf7\xf6\xb5){\n'\x06\x9en\xf1\x1e\xc7M\xec\xed\xaf\xd2Lm\xedk\xeb\x100\xe4\xbf\xef\xa0\xdf\xd6\xf2\x8dg?\x16\xcf?\x17\x84\x95m]\xf2\xfc\xf2\xc4\x8d\x1fm\x8c\xcc#\xfd\xe9\xb2W\xc2\xef\x85\xe0dI\x02\xc6\xd6\x00\x8c\xf3\xdf\xb9\xa9z\xf7\xef\xf4V\xbc\xd5~\xb4\xea>\x84\xe6\x05\xcei\xde@\xf5\xd3\x18\xb8\xde;J\xea\r\xc5\xfam-\x1fu\x87\xe9\xf5\xbe?;TFr`)i9\x1c\xf8\xd8\xaf\xa2\xc8\x9b_\xb5\xa7\xca\xe3_\xa0\xf1\xcf\x8a\xa3G\xf9\r\xa8\xbd\x010=J\xe2?\x9fJ\x1f\xef\xe0\x17]\xba\xea\xe2#\xd1\xfb\x07\ng\x9c\xdf\xb9\x10?\xf7]5\xafb\xdd\x96\x193\xb1\xfcw\xc7\x90\xb3p\x12\x04*\n\x19\x89-?\xf4\x01p'\xfd\xa9\xa5B\xeb\xc7\xaf\xba\xe8\xcb\xe9\xb3\xb1\xc5\xc1O\xe4\xac\xccm+@uu'\xda\xf2\x13\x86\x8f!\xe1O-O\x9c^FW\x96eK\xc3\xb3\x9b\xd6\x9b\x9f\x13\xe2N\xda\xe2\xf9&\xdeu\x0c=\xfb\xbc3\xa0(\xaai\x0bBuCcHw\x92\xfd)\x06m,\xad\xb2g\x1e\xb0\x06\xd0\xe9D\xeb\x8bV\xd6\xafK\xe9\x18\xfb\xcf,\xe4h\x8b\xc1b?\x9dFS?\x9d\x00\x98vm\xb6T\\\xc2~\x9e`\xad\xbfm\xdb\xf8\x96_\x1f\xec*\xa8ZSB\x9e\xfe:\xa4\xd2\xec\x93\xf8\xc8\x9f\xea\xa7v6\x0cS\xc0\xd8O\xff7qC\xf1k.\xf88\x87+\xf2O}\xdc\xf8\xdd\x97w\x16\xf3\r\xaa\x9f\xab\xf2\xbb\x9c\xf4\x92\xad\x89d6\xb6\x820`\xfdxk\xc8B\x9fK}x\x87\xac\xe8\xe7|\xc3V\xfa\xe5P\xf6\x0c\xe1\xee\x9a\x18^.\xdc\xb31~\xadg\x1154%\xd8/\x07*J\xfa\xab\xf9\x18\xd0~\xecf6\xa6\x1c\xeb&\x04,-\xda\x8c\x80t\x82\xbd\xe4\xf3\xc7\xf2\x91\x9b4\xa97\xc9\xee\xf6\xf0\\\x84N\xb3}>7\xd5\xb1\xec\xcf\x18\xc8\x03\xe4V\xe2\x1f\xf6{\"pv\x1aQ\xe1N\xda\xffx$\x93\xd3\x1eS\xed\r\x08\xf0Ln\x1bA\xd2\x1aB\xb9\xc7X=\xb9\xd6\x85>m\xd5\n\x9b\x16\x14\xf3\xdb\x06\x04\xd3\x80\xf5/\x97\xa5|g\xfdt[l3\xf7'wL\t\xb8^\xd9-\xeb\xef\xf2\xb8{\x1fF\x9e!4\x06p\x19wQ\x9f\xd9d\xa3\xcb\xd1N\x9c\x82\xd5\xf0\xca\xf1?\xb5\t\x08\xc5\x85\x1b\xef\x8a\xc3/:\x131\xde\xcaKA)a\xb7;&\xc8\xc2\xd2>c\x17\xb2 E\xbe\xce\xf7)(\x1d\x00\xbb\xcc\x04\xbc\x8cK\x89\x08U\xa6\x10)\xfb\xc2\x9e\xc0\xcb]\x17>\xe3G\xcd\xcbN\xa8\t\x9f88\x95\x8a\xae\x0fg\x9f\xf4\xaf\xce?T\xf7\xe0:\x0c\xe2\xcb\xff\xbb~\x85\x90M\xfc\x1f\x9b\x86  &\xa4\xbb\x99\x86\x90\x18t\xf0\x13\xcf\xab\x9a\xc9\x96\xbcM\xafq\x0f\"\x89\xf2R\"^\xb15\xab\x94\x87\xc8\t\xc5\x1e\xfa\x89(\xde\x05\xb0\x06\xba\xe2#\xbf\xfb\xd7\x17\xe6\xe4\xfa\xbeJ`\xe8w?\x94\xa7q\xf8 \xfdN\x96,\xca\xad\xb1\x07:H\x9d\x08\xd5[&\xf2H7\xa5\xab6\x84v\xa1\x9eA\x9a\xe5\x01:o\x18\xa4\xba\xe5\xdbj\x8ff\xd3\x98\xfc\xf5\xd2=\xfc.@\xa7\x9f\n5\xb8\n\xe3\xa0\x19_>\x02!\x01`\x13\xfbS\xcbi\xe6$\x13\x00\xe7\x10\n\x00E\xb7h\xfa9\x976\xbbiN\t\xad\x93\xfe2\xdf\xa7\xe3\xf3\x838\x83s\xa1`\xfe\xb7\x83\xa9\xc6\x87~\xfa>\\^\xc8!z\xab\xd89\xca\xb2\xd4\x14\xa1\xbb{>%(\xd1O\x97}6Y\xfbyN\xce\xd6y\xbf/\xd3\x9cK\xd5~\xf1`\xe8c\x88\x08J\xc4o\xf4+y_cBp\xd8G?\x9f\xf9\xf8\xacGt\xa9\x1e\xef\x906\xe5\xcc\x94\x03\x81D\x83\xd6\"\xa7\x05\xa5Bm\x90!\xf8\xf2\xa6AT\xef#\xfe\rA\xfe\xf6\x12\x83l:\xe1[\xa7\x046\xd4&\x9b?\x9d-M\x8a\xa8Bl\xcb\xf6\x11\xaaZN4\x0b|\xbe\x82v\x97\xa9\xd8g\xf6Wn\xd8\xb79\xa9R\xd3\xcc\xaf[\t\x99\x9c\xc1\xcf\xf4\x9bBz\xf0\xeb\xa7\xcf0\xefX\xc8\xf1\x1f\t|\xe4u\x98c\x83\x8eT\xe2\x82\xa6\x03\x9d\x0f\x0cz\xfd\xf4\xa8\xee\x8e\xf2i#\x11\xe5\xf84\x1d\xbf\xea\xc2\x92\xba\xaa\x92\x0e\x97%\x13\xf9\xdcTBk\x07;\x00e\x99f`\xa0\x1c\xf912LI\xcb\xfa\xfd&P<=J\x86\xa6\x08J\xd4\x16\xf2\x89\xb9x\x9b1\x81-q\xbe\xf4\xeb\x1e\x0e\xc9\xe6U\xd0\xc9\x9e\xa1\xa4H[\xe8\xa7\xeeO\xfc\xd6\x1c\x03|\xd5\xb6\xcc\\\x07z\xa9e;\ro\xc5\xce\x84\x15qo\xd5\xd8\x9cw\xc8\x83\x07\xbb\xf7\x9b\xfc\x19*\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x00\x10\x00\x02@\x00\x08\x00\x01 \x00\x04\x80\x80\x9d\t\xfc\x17b@\xfb."
            )
        elif kwargs["width"] == 10 and kwargs["height"] == 100:
            return BytesIO(
                b"II*\x00\x08\x00\x00\x00\x14\x00\x00\x01\x03\x00\x01\x00\x00\x00\n\x00\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00d\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\xfe\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\x06\x01\x00\x00\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00(\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00=\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00B\x01\x03\x00\x01\x00\x00\x00\x00\x01\x00\x00C\x01\x03\x00\x01\x00\x00\x00\x00\x01\x00\x00D\x01\x04\x00\x01\x00\x00\x00\xe6\x01\x00\x00E\x01\x04\x00\x01\x00\x00\x00P\x01\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\xd8\x85\x0c\x00\x10\x00\x00\x00\x0e\x01\x00\x00\xaf\x87\x03\x00 \x00\x00\x00\x8e\x01\x00\x00\xb0\x87\x0c\x00\x02\x00\x00\x00\xce\x01\x00\x00\xb1\x87\x02\x00\x08\x00\x00\x00\xde\x01\x00\x00\x00\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00G\x029\x9f\xa4\xa1\xe9?\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00J\x82\x8fv\xb0\xa9`\xc0\x00\x00\x00\x00\x00\x00\x00\x00\xf26`\xb3Gz\xd3?\x00\x00\x00\x00\x00\x00\x00\x00\x03\x9f\xa0>%z7@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x01\x00\x01\x00\x00\x00\x07\x00\x00\x04\x00\x00\x01\x00\x02\x00\x01\x04\x00\x00\x01\x00\x01\x00\x00\x08\x00\x00\x01\x00\xe6\x10\x01\x08\xb1\x87\x07\x00\x00\x00\x06\x08\x00\x00\x01\x00\x8e#\t\x08\xb0\x87\x01\x00\x01\x00\x0b\x08\xb0\x87\x01\x00\x00\x00\x88mt\x96\x1d\xa4r@\x00\x00\x00@\xa6TXAWGS 84|\x00x^\xed\xd6\xdb\t\x00 \x08\x00@m\xff\x91\x83\xa2!\x14\xa1s\x00\x1f\xa7\x1fF\x08\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 P'\x90q\xea\x92\xcbL\x80\x00\x01\x02\x04\x08\x10\x18(\x90\x9f\xbf?\xe6\x1fx\x94\x1d-\xbd\xc5\xaf\xcc\xddQK\r\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04\x08\x10 @\x80\x00\x01\x02\x04J\x05.\t^\x06\x01"
            )
        elif kwargs["width"] == 100 and kwargs["height"] == 2:
            return BytesIO(
                b"II*\x00\x08\x00\x00\x00\x14\x00\x00\x01\x03\x00\x01\x00\x00\x00d\x00\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\xfe\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\x06\x01\x00\x00\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00(\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00=\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00B\x01\x03\x00\x01\x00\x00\x00\x00\x01\x00\x00C\x01\x03\x00\x01\x00\x00\x00\x00\x01\x00\x00D\x01\x04\x00\x01\x00\x00\x00\xe6\x01\x00\x00E\x01\x04\x00\x01\x00\x00\x003\x01\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\xd8\x85\x0c\x00\x10\x00\x00\x00\x0e\x01\x00\x00\xaf\x87\x03\x00 \x00\x00\x00\x8e\x01\x00\x00\xb0\x87\x0c\x00\x02\x00\x00\x00\xce\x01\x00\x00\xb1\x87\x02\x00\x08\x00\x00\x00\xde\x01\x00\x00\x00\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x009\x8eZ\x0f\xe6\x84b?\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\xca?\xd1\xd4\xfb\x7ff\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00is2\x94c?\x00\x00\x00\x00\x00\x00\x00\x00\xd8K\xb6Z?\xfdK\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x01\x00\x01\x00\x00\x00\x07\x00\x00\x04\x00\x00\x01\x00\x02\x00\x01\x04\x00\x00\x01\x00\x01\x00\x00\x08\x00\x00\x01\x00\xe6\x10\x01\x08\xb1\x87\x07\x00\x00\x00\x06\x08\x00\x00\x01\x00\x8e#\t\x08\xb0\x87\x01\x00\x01\x00\x0b\x08\xb0\x87\x01\x00\x00\x00\x88mt\x96\x1d\xa4r@\x00\x00\x00@\xa6TXAWGS 84|\x00x^\xed\xd0\x81\x00\x00\x00\x00\x80\xa0\xfd\xa9\x17)\x84\n\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180P\x03\x00\x0f\x00\x01"
            )
        elif kwargs["width"] == 1 and kwargs["height"] == 1:
            return BytesIO(
                b"II*\x00\x08\x00\x00\x00\x15\x00\x00\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\n\x01\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\x12\x01\x00\x00\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00(\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00=\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00B\x01\x03\x00\x01\x00\x00\x00\x00\x01\x00\x00C\x01\x03\x00\x01\x00\x00\x00\x00\x01\x00\x00D\x01\x04\x00\x01\x00\x00\x00\xba\x01\x00\x00E\x01\x04\x00\x01\x00\x00\x003\x01\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x0e\x83\x0c\x00\x03\x00\x00\x00\x1a\x01\x00\x00\x82\x84\x0c\x00\x06\x00\x00\x002\x01\x00\x00\xaf\x87\x03\x00 \x00\x00\x00b\x01\x00\x00\xb0\x87\x0c\x00\x02\x00\x00\x00\xa2\x01\x00\x00\xb1\x87\x02\x00\x08\x00\x00\x00\xb2\x01\x00\x00\x00\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00\xa6\xf3\x16\xbf\x06\xd6\xdc?\xa6n\xf6\xcd]=\xc7?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xbd\xc5\xaf\x815\x87f\xc0\xfa\xbdu\xb2\xc1\x05U@\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x07\x00\x00\x04\x00\x00\x01\x00\x02\x00\x01\x04\x00\x00\x01\x00\x01\x00\x00\x08\x00\x00\x01\x00\xe6\x10\x01\x08\xb1\x87\x07\x00\x00\x00\x06\x08\x00\x00\x01\x00\x8e#\t\x08\xb0\x87\x01\x00\x01\x00\x0b\x08\xb0\x87\x01\x00\x00\x00\x88mt\x96\x1d\xa4r@\x00\x00\x00@\xa6TXAWGS 84|\x00x^\xed\xd0\x81\x00\x00\x00\x00\x80\xa0\xfd\xa9\x17)\x84\n\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180`\xc0\x80\x01\x03\x06\x0c\x180P\x03\x00\x0f\x00\x01"
            )


class MockWCS(WCS):
    """Test node that uses the MockClient above."""

    @property
    def client(self):
        return MockClient()

    def get_coordinates(self):
        return COORDS


class MockWCS(WCS):
    """Test node that uses the MockClient above, and injects podpac interpolation."""

    @property
    def client(self):
        return MockClient()

    def get_coordinates(self):
        return COORDS


class TestWCS(object):
    def test_eval_grid(self):
        c = COORDS

        node = MockWCS(source="mock", layer="mock")
        output = node.eval(c)
        assert output.shape == (100, 100)
        assert output.data.sum() == 1256581.0

    def test_eval_grid_chunked(self):
        c = COORDS

        node = MockWCS(source="mock", layer="mock", max_size=1000)
        output = node.eval(c)
        assert output.shape == (100, 100)
        assert output.data.sum() == 150.0

    def test_eval_grid_point(self):
        c = COORDS[50, 50]

        node = MockWCS(source="mock", layer="mock", max_size=1000)
        output = node.eval(c)
        assert output.shape == (1, 1)
        assert output.data.sum() == 0.0

    def test_eval_nonuniform(self):
        c = COORDS[[0, 10, 99], [0, 99]]

        node = MockWCS(source="mock", layer="mock")
        output = node.eval(c)
        assert output.shape == (100, 100)
        assert output.data.sum() == 1256581.0

    def test_eval_uniform_stacked(self):
        c = podpac.Coordinates([[COORDS["lat"], COORDS["lon"]]], dims=["lat_lon"])

        node = MockWCS(source="mock", layer="mock")
        output = node.eval(c)
        assert output.shape == (100,)
        # MPU Note: changed from 14350.0 to 12640.0 based on np.diag(node.eval(COORDS)).sum()
        assert output.data.sum() == 12640.0

    def test_eval_extra_unstacked_dim(self):
        c = podpac.Coordinates(["2020-01-01", COORDS["lat"], COORDS["lon"]], dims=["time", "lat", "lon"])

        node = MockWCS()
        output = node.eval(c)
        assert output.shape == (100, 100)
        assert output.data.sum() == 1256581.0

    def test_eval_extra_stacked_dim(self):
        c = podpac.Coordinates(
            [[COORDS["lat"][50], COORDS["lon"][50], 10]],
            dims=["lat_lon_alt"],
            crs="+proj=longlat +datum=WGS84 +no_defs +vunits=m",
        )

        node = MockWCS(source="mock", layer="mock", max_size=1000)
        output = node.eval(c)
        assert output.shape == (1,)
        assert output.data.sum() == 0.0

    def test_eval_missing_dim(self):
        c = podpac.Coordinates([COORDS["lat"]])

        node = MockWCS()
        with pytest.raises(ValueError, match="Cannot evaluate these coordinates"):
            output = node.eval(c)

    def test_eval_transpose(self):
        c = COORDS.transpose("lon", "lat")
        node = MockWCS(source="mock", layer="mock")
        output = node.eval(c)
        assert output.dims == ("lon", "lat")
        assert output.shape == (100, 100)
        assert output.data.sum() == 1256581.0

    def test_eval_other_crs(self):
        c = COORDS.transform("EPSG:3395")

        node = MockWCS()
        output = node.eval(c)
        assert output.shape == (100, 100)
        assert output.data.sum() == 1256581.0


class TestWCS(object):
    def test_eval_grid(self):
        c = COORDS

        node = MockWCS(source="mock", layer="mock").interpolate()
        output = node.eval(c)
        assert output.shape == (100, 100)
        assert output.data.sum() == 1256581.0

    def test_eval_nonuniform(self):
        c = COORDS[[0, 10, 99], [0, 99]]

        node = MockWCS(source="mock", layer="mock").interpolate()
        output = node.eval(c)
        assert output.shape == (3, 2)
        assert output.data.sum() == 0

    def test_eval_uniform_stacked(self):
        c = podpac.Coordinates([[COORDS["lat"], COORDS["lon"]]], dims=["lat_lon"])

        node = MockWCS(source="mock", layer="mock").interpolate()
        output = node.eval(c)
        assert output.shape == (100,)
        # MPU Note: changed from 14350.0 to 12640.0 based on np.diag(node.eval(COORDS)).sum()
        assert output.data.sum() == 12640.0


@pytest.mark.integration
class TestWCSIntegration(object):
    source = "https://maps.isric.org/mapserv?map=/map/sand.map"

    def setup_class(cls):
        cls.node1 = WCS(source=cls.source, layer="sand_0-5cm_mean", format="geotiff_byte", max_size=16384)
        cls.node2 = WCS(source=cls.source, layer="sand_0-5cm_mean", format="geotiff_byte", max_size=16384).interpolate()

    def test_coordinates(self):
        self.node1.coordinates

    def test_eval_grid(self):
        c = COORDS
        self.node1.eval(c)
        self.node2.eval(c)

    def test_eval_point(self):
        c = COORDS[50, 50]
        self.node1.eval(c)
        self.node2.eval(c)

    def test_eval_nonuniform(self):
        c = podpac.Coordinates([[-131.3, -131.4, -131.6], [23.0, 23.1, 23.3]], dims=["lon", "lat"])
        self.node1.eval(c)
        self.node2.eval(c)

    def test_eval_uniform_stacked(self):
        c = podpac.Coordinates([[COORDS["lat"][::4], COORDS["lon"][::4]]], dims=["lat_lon"])
        self.node1.eval(c)
        self.node2.eval(c)

    def test_eval_chunked(self):
        node = WCS(source=self.source, layer="sand_0-5cm_mean", format="geotiff_byte", max_size=4000)
        o1 = node.eval(COORDS)

    def test_eval_other_crs(self):
        c = COORDS.transform("EPSG:3395")
        self.node1.eval(c)
        self.node2.eval(c)

    def test_get_layers(self):
        # most basic
        layers = WCS.get_layers(self.source)
        assert isinstance(layers, list)

        # also works with nodes that have a builtin source
        class WCSWithSource(WCS):
            source = self.source

        layers = WCSWithSource.get_layers()
        assert isinstance(layers, list)
