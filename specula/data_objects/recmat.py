from iteras.data_objects.matrix import Matrix, matrix_array_name
from iteras.lib import storage


class Recmat(Matrix):
    """
    Reconstruction matrix data object.
    This class holds the information about the reconstruction matrix,
    which maps slopes to modes. The reconstruction matrix axes are [modes, slopes].

    It is a :class:`~iteras.data_objects.matrix.Matrix`, whose matrix is also
    available as the ``recmat`` attribute.
    """
    def __init__(self,
                 recmat,
                 modes2recLayer=None,  # TODO not used
                 norm_factor: float=0,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Initialize a :class:`~specula.data_objects.recmat.Recmat` object.
        """
        super().__init__(recmat, target_device_idx=target_device_idx, precision=precision)
        self.norm_factor = norm_factor
        self.proj_list = []
        self.modes2recLayer = None
#        self.set_modes2recLayer(modes2recLayer) # TODO

    @property
    def recmat(self):
        '''The reconstruction matrix, same as ``matrix``'''
        return self.matrix

    @recmat.setter
    def recmat(self, value):
        self.matrix = value

    def set_modes2recLayer(self, modes2recLayer):
        if modes2recLayer is not None:
            modes2recLayer = self.to_xp(modes2recLayer)
            n = modes2recLayer.shape
            for i in range(n[0]):
                idx = self.xp.where(modes2recLayer[i, :] > 0)[0]
                proj = self.xp.zeros((n[1], len(idx)), dtype=self.dtype)
                proj[idx, :] = self.xp.identity(len(idx))
                self.proj_list.append(proj)
            self.modes2recLayer = modes2recLayer

    def reduce_size(self, nModesToBeDiscarded):
        if nModesToBeDiscarded >= self.nmodes:
            raise ValueError(f"nModesToBeDiscarded should be less than nmodes (<{self.nmodes})")
        self.recmat = self.recmat[:self.nmodes - nModesToBeDiscarded, :]

    def get_fits_header(self):
        hdr = super().get_fits_header()
        hdr['NORMFACT'] = self.norm_factor
        return hdr

    @property
    def nmodes(self):
        return self.nrows

    def _arrays_to_save(self):
        return {'matrix': self.matrix, 'modes2reclayer': self.modes2recLayer}

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            hdr = f.header
            version = int(hdr['VERSION'])
            if version != 1:
                raise ValueError(f"Error: unknown version {version} in file {filename}")
            norm_factor = float(hdr['NORMFACT'])
            if f.has('modes2reclayer', 2):
                mode2reLayer = f.read('modes2reclayer', 2)
            else:
                mode2reLayer = None
            recmat = f.load(matrix_array_name(f), 1, target_device_idx=target_device_idx)
        return Recmat(recmat, mode2reLayer, norm_factor, target_device_idx=target_device_idx)
