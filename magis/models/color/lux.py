from .. import Model, Component, DualBoundaries, CircularBoundaries


class Lux(Model):
    '''
        LUX : The lexicon of uncertain color standards model
        -----------------------------------------------------
    
        lux.predict(datum) s.t. datum is (h,s,v), 
                                h scaled to (-180,180) or (0,360)
                                s scaled to (0, 100)
                                v scaled to (0, 100)
        lux.likelihood(datum, color_name)
        lux.posterior(datum)            
    '''
    def __init__(self, name, *args, **kwargs):
            
        super(Lux, self).__init__(name, *args, **kwargs)

        self.__dict__.update({c.name.replace(" ", "_").replace("-","_"):c 
                              for c in self.components})

    @classmethod
    def from_json(cls, filename, *args, **kwargs):
        return super(Lux, cls).from_json(filename, ColorLabel)

    @classmethod
    def pretrained(cls):
        import os
        HERE = os.path.dirname(os.path.abspath(__file__))
        return cls.from_json(os.path.join(HERE, 'assets', 'lux.json'))

class ColorLabel(Component):
    def __init__(self, name, dim_models, availability, *args, **kwargs):
        super(ColorLabel, self).__init__(name, *args, **kwargs)
        self.hue_model, self.sat_model, self.val_model = dim_models
        self.availability = availability

    @classmethod
    def from_dict(cls, info):
        p_order = ['scalelower', 'shapelower', 'mulower', 'scaleupper', 'shapeupper', 'muupper']
        
        prminfo = info['parameters'] 
        avail = prminfo['availability']
        h_prm, s_prm, v_prm = prminfo['parameters']
        h_adjust = prminfo['hue_adjust']
        hue_model = CircularBoundaries.from_parameters(*tuple([h_prm[p] for p in p_order] + [h_adjust]))
        sat_model = DualBoundaries.from_parameters(*tuple([s_prm[p] for p in p_order]))
        val_model = DualBoundaries.from_parameters(*tuple([v_prm[p] for p in p_order]))
    
        return cls(info['name'], (hue_model, sat_model, val_model), avail)

    def pdf(self, x):
        try: 
            return self.hue_model(x[:,0]) * self.sat_model(x[:,1]) * self.val_model(x[:,2])
        except Exception as e: # fix to specific one...  value error?
            return self.hue_model(x[0]) * self.sat_model(x[1]) * self.val_model(x[2])

    @property
    def prior(self):
        return self.availability



