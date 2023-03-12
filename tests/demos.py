from math import exp

import demes
import momi as momi2
import numpy as np


class SingleDeme:
    class Constant:
        def __init__(self, size=1.0):
            model1 = momi2.DemographicModel(N_e=size, muts_per_gen=1)
            model1.add_leaf("A", N=size)
            self.model1 = model1

            b = demes.Builder(description="demo")
            b.add_deme("A", epochs=[dict(start_size=size)])
            self.b = b

        def base(self):
            model1 = self.model1

            b = self.b
            g = b.resolve()

            return g, model1

    class Exponential:
        # Single deme, exponential population size
        def __init__(self, size=1.0, t=1.0, g=1.0):
            size_end = size * exp(-g * t)
            model1 = momi2.DemographicModel(N_e=size_end, muts_per_gen=1)
            model1.add_leaf("A", g=g, N=size)
            model1.set_size("A", g=0.0, t=t)
            self.model1 = model1

            b = demes.Builder(description="demo")
            b.add_deme(
                "A",
                epochs=[
                    dict(end_time=t, start_size=size_end),
                    dict(
                        end_time=0,
                        start_size=size_end,
                        end_size=size,
                        size_function="exponential",
                    ),
                ],
            )
            self.b = b

        def base(self):
            model1 = self.model1

            b = self.b
            g = b.resolve()

            return g, model1


class TwoDemes:
    class Constant:
        # Two demes, Constant population sizes
        def __init__(self, size=1.0, t=1.0):
            # momi2 model
            model1 = momi2.DemographicModel(N_e=size, muts_per_gen=1)
            model1.add_leaf("A", N=size)
            model1.add_leaf("B", N=size)
            model1.move_lineages("B", "A", t=t, N=size)
            self.model1 = model1

            b = demes.Builder()
            b.add_deme("AB", epochs=[dict(start_size=size, end_time=t)])
            b.add_deme("A", epochs=[dict(start_size=size)], ancestors=["AB"])
            b.add_deme("B", epochs=[dict(start_size=size)], ancestors=["AB"])
            self.b = b

            self.t = t

        def base(self):
            model1 = self.model1

            b = self.b
            g = b.resolve()

            return g, model1

        def pulse(self, tp=0.5, p=0.1):
            model1 = self.model1
            model1.move_lineages("B", "A", t=tp, p=p)

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=tp, proportions=[p])
            g = b.resolve()

            return g, model1

        def two_pulses(self, tp1=0.25, tp2=0.75, p1=0.1, p2=0.1):
            model1 = self.model1
            model1.move_lineages("B", "A", t=tp1, p=p1)
            model1.move_lineages("A", "B", t=tp2, p=p2)

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=tp1, proportions=[p1])
            b.add_pulse(sources=["B"], dest="A", time=tp2, proportions=[p2])
            g = b.resolve()

            return g, model1

        def five_pulses(self, p=0.1):
            t = self.t
            ts = [i * t / 6 for i in range(1, 6)]

            model1 = self.model1
            model1.move_lineages("B", "A", t=ts[0], p=p)
            model1.move_lineages("A", "B", t=ts[1], p=p)
            model1.move_lineages("A", "B", t=ts[2], p=p)
            model1.move_lineages("B", "A", t=ts[3], p=p)
            model1.move_lineages("A", "B", t=ts[4], p=p)

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=ts[0], proportions=[p])
            b.add_pulse(sources=["B"], dest="A", time=ts[1], proportions=[p])
            b.add_pulse(sources=["B"], dest="A", time=ts[2], proportions=[p])
            b.add_pulse(sources=["A"], dest="B", time=ts[3], proportions=[p])
            b.add_pulse(sources=["B"], dest="A", time=ts[4], proportions=[p])
            g = b.resolve()

            return g, model1

        def migration(self, tstart=1.0, tend=0.0, rate=0.05):
            model1 = None

            b = self.b
            b.add_migration(
                source="B", dest="A", start_time=tstart, end_time=tend, rate=rate
            )
            g = b.resolve()

            return g, model1

        def migration_twophase(self, tstart=1.0, tend=0.0, rate=0.05):
            model1 = None
            b = self.b
            b.add_migration(
                source="B", dest="A", start_time=tstart / 2, end_time=tend, rate=rate
            )
            b.add_migration(
                source="B", dest="A", start_time=tstart, end_time=tstart / 2, rate=rate
            )
            g = b.resolve()
            return g, model1

        def migration_sym(self, tstart=1, tend=0.0, rate=0.05):
            model1 = None

            b = self.b
            b.add_migration(
                demes=["A", "B"], start_time=tstart, end_time=tend, rate=rate
            )
            g = b.resolve()

            return g, model1

        def migration_sym_pulse(self, tp=0.5, p=0.1, tstart=1, tend=0.0, rate=0.05):
            model1 = None

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=tp, proportions=[p])
            b.add_migration(
                demes=["A", "B"], start_time=tstart, end_time=tend, rate=rate
            )
            g = b.resolve()

            return g, model1

    class Exponential:
        def __init__(self, size=1, t=1.0, g=1.0, size_scale=1.25):
            # Two demes, exponential population size, no migration
            tgA = t / 2
            tgB = t / 3
            gA = 2 * g
            gB = g
            sizebottomA = size
            sizebottomB = size_scale * size
            sizetopA = sizebottomA * exp(-gA * tgA)
            sizetopB = sizebottomB * exp(-gB * tgB)

            model1 = momi2.DemographicModel(N_e=size, muts_per_gen=1)
            model1.add_leaf("A", g=gA, N=sizebottomA)
            model1.set_size("A", g=0.0, t=tgA, N=sizetopA)
            model1.add_leaf("B", g=gB, N=sizebottomB)
            model1.set_size("B", g=0.0, t=tgB, N=sizetopB)
            model1.move_lineages("B", "A", t=t, N=size, g=0.0)
            self.model1 = model1

            b = demes.Builder(description="demo")
            b.add_deme("AB", epochs=[dict(end_time=t, start_size=size)])
            b.add_deme(
                "A",
                ancestors=["AB"],
                epochs=[
                    dict(end_time=tgA, start_size=sizetopA),
                    dict(
                        end_time=0,
                        start_size=sizetopA,
                        end_size=sizebottomA,
                        size_function="exponential",
                    ),
                ],
            )
            b.add_deme(
                "B",
                ancestors=["AB"],
                epochs=[
                    dict(end_time=tgB, start_size=sizetopB),
                    dict(
                        end_time=0,
                        start_size=sizetopB,
                        end_size=sizebottomB,
                        size_function="exponential",
                    ),
                ],
            )
            self.b = b

        def base(self):
            model1 = self.model1

            b = self.b
            g = b.resolve()

            return g, model1

        def pulse(self, tp=0.25, p=0.1):
            model1 = self.model1
            model1.move_lineages("B", "A", t=tp, p=p)

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=tp, proportions=[p])
            g = b.resolve()

            return g, model1

        def two_pulses(self, tp1=0.25, tp2=0.75, p1=0.1, p2=0.1):
            model1 = self.model1
            model1.move_lineages("B", "A", t=tp1, p=p1)
            model1.move_lineages("A", "B", t=tp2, p=p2)

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=tp1, proportions=[p1])
            b.add_pulse(sources=["B"], dest="A", time=tp2, proportions=[p2])
            g = b.resolve()

            return g, model1

        def migration(self, tstart=1, tend=0.0, rate=0.05):
            model1 = None

            b = self.b
            b.add_migration(
                source="B", dest="A", start_time=tstart, end_time=tend, rate=rate
            )
            g = b.resolve()

            return g, model1

        def migration_sym(self, tstart=1, tend=0.0, rate=0.05):
            model1 = None

            b = self.b
            b.add_migration(
                demes=["A", "B"], start_time=tstart, end_time=tend, rate=rate
            )
            g = b.resolve()

            return g, model1

        def pulse_migration(self, tp=0.5, p=0.1, tstart=1.0, tend=0.0, rate=0.05):
            model1 = None

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=tp, proportions=[p])
            b.add_migration(
                source="B", dest="A", start_time=tstart, end_time=tend, rate=rate
            )
            g = b.resolve()

            return g, model1


class ThreeDemes:
    class Constant:
        # Three demes, Constant Size
        def __init__(self, size=1.0, t1=1.0, t2=2.0):
            model1 = momi2.DemographicModel(N_e=size, muts_per_gen=1)
            model1.add_leaf("A", N=size)
            model1.add_leaf("B", N=size)
            model1.add_leaf("C", N=size)
            model1.move_lineages("B", "A", t=t1, N=size)
            model1.move_lineages("C", "A", t=t2, N=size)
            self.model1 = model1

            b = demes.Builder()
            b.add_deme("ABC", epochs=[dict(start_size=size, end_time=t2)])
            b.add_deme(
                "AB", ancestors=["ABC"], epochs=[dict(start_size=size, end_time=t1)]
            )
            b.add_deme("A", epochs=[dict(start_size=size)], ancestors=["AB"])
            b.add_deme("B", epochs=[dict(start_size=size)], ancestors=["AB"])
            b.add_deme("C", epochs=[dict(start_size=size)], ancestors=["ABC"])
            self.b = b

            self.size = size

        def base(self):
            model1 = self.model1

            b = self.b
            g = b.resolve()

            return g, model1

        def migration(self, rate=0.05):
            model1 = None

            b = self.b
            b.add_migration(source="C", dest="AB", rate=rate)
            g = b.resolve()

            return g, model1

        def pulses(self, npulses=5, p=0.1):
            b = self.b
            model1 = self.model1

            np.random.seed(108)
            for _ in range(npulses):
                source, dest = np.random.choice(["A", "B", "C"], 2, replace=False)
                source = str(source)
                dest = str(dest)
                tp = np.random.rand()

                b.add_pulse(sources=[source], dest=dest, time=tp, proportions=[p])
                model1.move_lineages(dest, source, t=tp, p=p)

            g = b.resolve()

            return g, model1

        def pulses_migration(self, tp1=0.25, p1=0.1, tp2=0.75, p2=0.1, rate=0.05):
            model1 = None

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=tp1, proportions=[p1])
            b.add_pulse(sources=["B"], dest="C", time=tp2, proportions=[p2])
            b.add_migration(source="C", dest="AB", rate=rate)
            g = b.resolve()

            return g, model1

        def three_migrants(self, rate):
            model1 = None

            b = self.b
            b.add_migration(demes=["A", "B", "C"], rate=rate)
            g = b.resolve()

            return g, model1

    class Exponential:
        # Three demes, exponential growth
        def __init__(self, size=1.0, t=1, g=1.0):
            tgA = t / 2
            tgB = t / 3
            gA = 2 * g
            gB = g
            sizebottomA = size
            sizebottomB = 1.25 * size
            sizetopA = sizebottomA * exp(-gA * tgA)
            sizetopB = sizebottomB * exp(-gB * tgB)

            model1 = momi2.DemographicModel(N_e=size, muts_per_gen=1)
            model1.add_leaf("A", g=gA, N=sizebottomA)
            model1.set_size("A", g=0.0, t=tgA, N=sizetopA)
            model1.add_leaf("B", g=gB, N=sizebottomB)
            model1.set_size("B", g=0.0, t=tgB, N=sizetopB)
            model1.add_leaf("C", N=size)
            model1.move_lineages("B", "A", t=t, N=size, g=0.0)
            model1.move_lineages("C", "A", t=2 * t, N=size, g=0.0)
            self.model1 = model1

            b = demes.Builder(description="demo")
            b.add_deme("ABC", epochs=[dict(end_time=2 * t, start_size=size)])
            b.add_deme(
                "AB", ancestors=["ABC"], epochs=[dict(end_time=t, start_size=size)]
            )
            b.add_deme(
                "A",
                ancestors=["AB"],
                epochs=[
                    dict(end_time=tgA, start_size=sizetopA),
                    dict(
                        end_time=0,
                        start_size=sizetopA,
                        end_size=sizebottomA,
                        size_function="exponential",
                    ),
                ],
            )
            b.add_deme(
                "B",
                ancestors=["AB"],
                epochs=[
                    dict(end_time=tgB, start_size=sizetopB),
                    dict(
                        end_time=0,
                        start_size=sizetopB,
                        end_size=sizebottomB,
                        size_function="exponential",
                    ),
                ],
            )
            b.add_deme("C", ancestors=["ABC"], epochs=[dict(start_size=size)])
            self.b = b

        def base(self):
            model1 = self.model1

            b = self.b
            g = b.resolve()

            return g, model1

        def migration(self, rate=0.05):
            model1 = None

            b = self.b
            b.add_migration(source="C", dest="AB", rate=rate)
            g = b.resolve()

            return g, model1

        def migrations(self, rate=0.05):
            model1 = None

            b = self.b
            b.add_migration(demes=["A", "B", "C"], rate=rate)
            g = b.resolve()

            return g, model1

        def pulses_migration(self, tp1=0.25, p1=0.1, tp2=0.75, p2=0.1, rate=0.05):
            model1 = None

            b = self.b
            b.add_pulse(sources=["A"], dest="B", time=tp1, proportions=[p1])
            b.add_pulse(sources=["B"], dest="C", time=tp2, proportions=[p2])
            b.add_migration(source="C", dest="AB", rate=rate)
            g = b.resolve()

            return g, model1


class MultiAnc:
    # Three demes, multiple ancestry
    def __init__(self, size=1.0):
        # Multi Ancestry is not implemented in momi2
        self.model1 = None

        b = demes.Builder()
        b.add_deme("C", epochs=[dict(start_size=size, end_time=0.5)])
        b.add_deme("B", epochs=[dict(start_size=size)], ancestors=["C"], start_time=1.5)
        b.add_deme(
            "A",
            epochs=[dict(start_size=size)],
            start_time=1.0,
            ancestors=["C", "B"],
            proportions=[0.3, 0.7],
        )
        self.b = b

    def base(self):
        model1 = self.model1

        b = self.b
        g = b.resolve()

        return g, model1


class FiveDemes:
    # Five Demes
    def __init__(self, size=1.0, t=1.0):
        # 5 demes, 2 pulses
        np.random.seed(108)
        Ne1, Ne2, Ne3, Ne4, Ne5, Ne12, Ne123, Ne45, Ne12345 = size * np.random.rand(9)
        t12, t123, t45, t12345 = [t / 2, t, 2 * t / 3, 2 * t]

        model1 = momi2.DemographicModel(N_e=Ne12345, muts_per_gen=1)
        model1.add_leaf("A", N=Ne1)
        model1.add_leaf("B", N=Ne2)
        model1.add_leaf("C", N=Ne3)
        model1.add_leaf("D", N=Ne4)
        model1.add_leaf("E", N=Ne5)
        model1.move_lineages("B", "A", t=t12, N=Ne12)
        model1.move_lineages("C", "A", t=t123, N=Ne123)
        model1.move_lineages("E", "D", t=t45, N=Ne45)
        model1.move_lineages("D", "A", t=t12345, N=Ne12345)
        self.model1 = model1

        b = demes.Builder(description="demo")
        b.add_deme("ABCDE", epochs=[dict(end_time=t12345, start_size=Ne12345)])
        b.add_deme(
            "ABC", ancestors=["ABCDE"], epochs=[dict(end_time=t123, start_size=Ne123)]
        )
        b.add_deme(
            "AB", ancestors=["ABC"], epochs=[dict(end_time=t12, start_size=Ne12)]
        )
        b.add_deme("A", ancestors=["AB"], epochs=[dict(end_time=0, start_size=Ne1)])
        b.add_deme("B", ancestors=["AB"], epochs=[dict(end_time=0, start_size=Ne2)])
        b.add_deme("C", ancestors=["ABC"], epochs=[dict(end_time=0, start_size=Ne3)])
        b.add_deme(
            "DE", ancestors=["ABCDE"], epochs=[dict(end_time=t45, start_size=Ne45)]
        )
        b.add_deme("D", ancestors=["DE"], epochs=[dict(end_time=0, start_size=Ne4)])
        b.add_deme("E", ancestors=["DE"], epochs=[dict(end_time=0.0, start_size=Ne5)])
        self.b = b

    def base(self):
        model1 = self.model1

        b = self.b
        g = b.resolve()

        return g, model1

    def pulses(self, tp1=0.25, tp2=1.25, p1=0.1, p2=0.1):
        model1 = self.model1
        model1.move_lineages("B", "A", t=tp1, p=p1)
        model1.move_lineages("D", "A", t=tp2, p=p2)

        b = self.b
        b.add_pulse(sources=["A"], dest="B", proportions=[p1], time=tp1)
        b.add_pulse(sources=["ABC"], dest="DE", proportions=[p2], time=tp2)
        g = b.resolve()

        return g, model1


class Experimental:
    class TwoDemesTwoEpochs:
        def __init__(self, size=1.0, t=1.0, epoch_time=0.5):
            b = demes.Builder()
            b.add_deme("AB", epochs=[dict(start_size=size, end_time=t)])
            b.add_deme(
                "A",
                epochs=[
                    dict(start_size=size, end_time=epoch_time),
                    dict(start_size=size, end_time=0.0),
                ],
                ancestors=["AB"],
            )
            b.add_deme(
                "B",
                epochs=[
                    dict(start_size=size, end_time=epoch_time),
                    dict(start_size=size, end_time=0.0),
                ],
                ancestors=["AB"],
            )
            self.b = b
            self.model1 = None
            self.epoch_time = 0.5
            self.t = t

        def base(self):
            model1 = self.model1

            b = self.b
            g = b.resolve()

            return g, model1

        def migrations(self, rate=0.05):
            model1 = self.model1
            t = self.t
            b = self.b
            epoch_time = self.epoch_time

            b.add_migration(
                demes=["A", "B"], start_time=t, end_time=epoch_time, rate=rate
            )
            b.add_migration(
                demes=["A", "B"], start_time=epoch_time, end_time=0, rate=rate
            )

            b = self.b
            g = b.resolve()

            return g, model1
