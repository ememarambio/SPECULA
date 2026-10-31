from astropy.io import fits

from specula.data_objects.electric_field import ElectricField
from iteras.lib import storage

class Layer(ElectricField):
    """
    Layer data object.
    This object represents a layer in the atmosphere for wavefront propagation simulations.
    It inherits from the ElectricField class and adds additional properties such as height,
    shifts, rotation, and magnification.
    """
    def __init__(self,
                 dimx: int,
                 dimy: int,
                 pixel_pitch: float,
                 height: float,
                 shiftXYinPixel: tuple=(0.0, 0.0),
                 rotInDeg: float=0.0,
                 magnification: float=1.0,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Initialize a :class:`~specula.data_objects.layer.Layer` object.

        Parameters
        ----------
        dimx : int [pixels]
            Number of pixels along the x-axis (width).
        dimy : int [pixels]
            Number of pixels along the y-axis (height).
        pixel_pitch : float [m]
            The dimension in meters of a pixel.
        height : float [m]
            The height of the layer.
        shiftXYinPixel : tuple [pixels], optional
            The (x, y) shift of the layer in pixel units (default: (0.0, 0.0)).
        rotInDeg : float [deg], optional
            The rotation of the layer in degrees (default: 0.0).
        magnification : float [1], optional
            The magnification factor of the layer (default: 1.0).
        target_device_idx : int [1], optional
            Device index for computation (default: None).
        precision : int [1], optional
            Precision for computation (default: None).
        """
        super().__init__(dimx, dimy, pixel_pitch, target_device_idx=target_device_idx, precision=precision)
        self.height = height
        self.shiftXYinPixel = self.to_xp(shiftXYinPixel).astype(self.dtype)
        self.rotInDeg = rotInDeg
        self.magnification = magnification

    # get_value and set_value are inherited from ElectricField

    def debug_summary(self):
        return self.phaseInNm

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Layer'
        hdr['DIMX'] = self.field[0].shape[1]
        hdr['DIMY'] = self.field[0].shape[0]
        hdr['PIXPITCH'] = self.pixel_pitch
        hdr['HEIGHT'] = self.height
        hdr['SHIFTX'] = float(self.shiftXYinPixel[0])
        hdr['SHIFTY'] = float(self.shiftXYinPixel[1])
        hdr['ROTATION'] = self.rotInDeg
        hdr['MAGNIFIC'] = self.magnification
        return hdr

    def save(self, filename, overwrite=True):
        storage.write_object(filename, self.get_fits_header(),
                             {'amplitude': self.field[0], 'phase': self.field[1]},
                             overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        dimx = int(hdr['DIMX'])
        dimy = int(hdr['DIMY'])
        pitch = float(hdr['PIXPITCH'])
        height = float(hdr['HEIGHT'])
        shiftX = float(hdr['SHIFTX'])
        shiftY = float(hdr['SHIFTY'])
        rotInDeg = float(hdr['ROTATION'])
        magnification = float(hdr['MAGNIFIC'])
        layer = Layer(dimx, dimy, pitch, height, (shiftX, shiftY), rotInDeg, magnification, target_device_idx=target_device_idx)
        return layer

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            hdr = f.header
            if 'OBJ_TYPE' not in hdr or hdr['OBJ_TYPE'] != 'Layer':
                raise ValueError(f"Error: file {filename} does not contain a Layer object")
            obj = Layer.from_header(hdr, target_device_idx=target_device_idx)
            obj.field[0, :] = obj.to_xp(f.read('amplitude', 1))
            obj.field[1, :] = obj.to_xp(f.read('phase', 2))
        return obj

    # array_for_display is inherited from ElectricField
