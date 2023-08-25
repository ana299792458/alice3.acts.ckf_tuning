#!/usr/bin/env python3
from pathlib import Path
from typing import Optional, Union
from collections import namedtuple
import argparse
import sys
import os
import pathlib
import numpy as np  

from acts.examples import Sequencer, GenericDetector, RootParticleReader
from acts.examples.simulation import ParticleSelectorConfig
from acts.examples.reconstruction import TrackSelectorConfig

sys.path.append('/Users/achalume/alice/actsdir/source/Examples/Scripts/Python')
from alice3 import buildALICE3Geometry

import acts

from acts import UnitConstants as u


def getArgumentParser():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Command line arguments for CKF")
    parser.add_argument(
        "-i",
        "--indir",
        type=Path,
        dest="indir",
        help="Directory with input root files",
        default="~/alice/actsdir/test-ckf/input/",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="outdir",
        help="Output directory for new ntuples",
        default="./",
    )

    parser.add_argument(
        "-g",
        "--geodir"
        "geo_dir",
        type=Path,
        dest="geo_dir",
        help="Input directory containing the ALICE3 standalone geometry.",
        default="/Users/achalume/alice/actsdir/"
    )

    parser.add_argument(
        "-n", "--nEvents", dest="nEvts", help="Number of events to run over", default=10
    )
    parser.add_argument(
        "--sf_maxSeedsPerSpM",
        dest="sf_maxSeedsPerSpM",
        help="Number of compatible seeds considered for middle seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sf_cotThetaMax",
        dest="sf_cotThetaMax",
        help="cot of maximum theta angle",
        type=float,
        default=27.2899,
    )
    parser.add_argument(
        "--sf_sigmaScattering",
        dest="sf_sigmaScattering",
        help="How many sigmas of scattering to include in seeds",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--sf_radLengthPerSeed",
        dest="sf_radLengthPerSeed",
        help="Average Radiation Length",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--sf_impactMax",
        dest="sf_impactMax",
        help="max impact parameter in mm",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--sf_maxPtScattering",
        dest="sf_maxPtScattering",
        help="maximum Pt for scattering cut in GeV",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--sf_deltaRMin",
        dest="sf_deltaRMin",
        help="minimum value for deltaR separation in mm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sf_deltaRMax",
        dest="sf_deltaRMax",
        help="maximum value for deltaR separation in mm",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--no-material", 
        action="store_true", 
        help="Decorate material to the geometry"
    )

    parser.add_argument(
    "--etaMin", 
    dest="etaMin", 
    help="minimum value for eta",
    type=float,
    default=-4.0
    )

    parser.add_argument(
    "--etaMax", 
    dest="etaMax", 
    help="maximum value for eta",
    type=float,
    default=4.0
    )

    return parser


def runCKFTracks(
    trackingGeometry,
    decorators,
    geometrySelection: Path,
    digiConfigFile: Path,
    field,
    outputDir: Path,
    NumEvents=10,
    EtaMin=-4.0,
    EtaMax=4.0,
    truthSmearedSeeded=False,
    truthEstimatedSeeded=False,
    outputCsv=True,
    inputParticlePath: Optional[Path] = None,
    s=None,
    MaxSeedsPerSpM=1,
    CotThetaMax=27.2899,
    SigmaScattering=5.0,
    RadLengthPerSeed=0.1,
    ImpactMax=3.0,
    MaxPtScattering=100.0,
    DeltaRMin=1.0,
    DeltaRMax=60.0,
):
    #outputDir = outputDir
    npart = 10
    nEvents = NumEvents
    ptMin = 100.0
    EtaMinMin = min(abs(EtaMin),abs(EtaMax))
    pMin = ptMin*np.cosh(EtaMinMin)
    nHits = 7
    nMeasCutOff = 1
    seedConf = False

    # outputDir = pathlib.Path(
    # "/Users/achalume/alice/actsdir/acts/bin/output/python/bin_search/ckfC_" +
    # "13" + "_npart" +
    # str(npart) + "_nEvents" + str(nEvents) +
    # "_noambi_noinfl_nhits"+ str(nHits) +"_binsauto")#pTpmin"+str(int(ptMin))) #+"_pMin"+str(int(pMin)))

    from acts.examples.simulation import (
        addParticleGun,
        MomentumConfig,
        EtaConfig,
        PhiConfig,
        ParticleConfig,
        addFatras,
        addDigitization,
    )

    from acts.examples.reconstruction import (
        addSeeding,
        TruthSeedRanges,
        ParticleSmearingSigmas,
        SeedFinderConfigArg,
        SeedFinderOptionsArg,
        SeedingAlgorithm,
        TruthEstimatedSeedingAlgorithmConfigArg,
        addCKFTracks,
        SeedFilterConfigArg, 
        SpacePointGridConfigArg,
        addAmbiguityResolution,
        AmbiguityResolutionConfig,
        SeedingAlgorithmConfigArg,
    )

    s = s or acts.examples.Sequencer(
        events=int(NumEvents),
        numThreads=15, #-1 for multiple thread modes.
        logLevel=acts.logging.INFO,
        outputDir=str(outputDir),
    )
    for d in decorators:
        s.addContextDecorator(d)
    rnd = acts.examples.RandomNumbers(seed=42)
    outputDir = Path(outputDir)

    if inputParticlePath is None:
        print('\nPARTICLE GUN USED\n')
        addParticleGun(
            s,
            MomentumConfig(ptMin * u.MeV, 10.0 * u.GeV, transverse=True),
            EtaConfig(EtaMin, EtaMax, uniform=True),
            ParticleConfig(npart, acts.PdgParticle.eMuon, True), #4
            PhiConfig(0.0, 360.0 * u.degree),
            multiplicity=1, #2
            rnd=rnd,
        )
    else:
        acts.logging.getLogger("CKFExample").info(
            "Reading particles from %s", inputParticlePath.resolve()
        )
        assert inputParticlePath.exists()
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                particleCollection="particles_input",
                orderedEvents=False,
            )
        )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        preSelectParticles=ParticleSelectorConfig(
            eta=(-4.0, 4.0), pt=(ptMin * u.MeV, None), removeNeutral=True),
        pMin=pMin * u.MeV,
        enableInteractions=True,
        outputDirRoot=outputDir,
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        TruthSeedRanges(pt=(ptMin * u.MeV, None), eta=(EtaMin,EtaMax), nHits=(nHits, None)),
        ParticleSmearingSigmas(pRel=0.01),  # only used by SeedingAlgorithm.TruthSmeared
        SeedFinderConfigArg(
            r=(None, 200 * u.mm),  # rMin=default, 33mm
            deltaR=(DeltaRMin * u.mm, DeltaRMax * u.mm),
            collisionRegion=(-250 * u.mm, 250 * u.mm),
            z=(-1300 * u.mm, 1300 * u.mm),
            maxSeedsPerSpM=MaxSeedsPerSpM,
            cotThetaMax=CotThetaMax,
            sigmaScattering=SigmaScattering,
            radLengthPerSeed=RadLengthPerSeed,
            maxPtScattering=MaxPtScattering * u.GeV,
            minPt=ptMin * u.MeV,
            impactMax=ImpactMax * u.mm,
            # could add also: 
            # zOutermostLayers
            # binSizeR = ? 1 * u.mm
            # zBinsCustomLooping ?
            # deltaRMaxTopSP, deltaRMaxBottomSP, deltaZMax ?

            # rRangeMiddleSP=[
            #     [1,75],
            #     [1,100],
            #     [1,100],
            #     [4,150],
            #     [4,150],
            #     [4,150],
            #     [4,150],
            #     [4,150],
            #     [4,150],
            #     [4,150],
            #     [1,100],
            #     [1,100],
            #     [1,75],
            # ],

            seedConfirmation=seedConf,
            useVariableMiddleSPRange=True,
             #deltaRMiddleMinSPRange=10 * u.mm,
             #deltaRMiddleMaxSPRange=10 * u.mm,
            deltaRMiddleSPRange=(1 * u.mm, 10 * u.mm),

        ),

        SeedFinderOptionsArg(bFieldInZ=2 * u.T, beamPos=(0.0, 0, 0)), #was 1.99724
        
        SeedFilterConfigArg(
            # impactWeightFactor=100,
            # zOriginWeightFactor=1,
            # compatSeedWeight=3,
            # compatSeedLimit=3,

            maxSeedsPerSpMConf=5,
            seedConfirmation=seedConf,
            maxQualitySeedsPerSpMConf=5, #for seedConf
        ),

        SpacePointGridConfigArg(
            # zBinEdges=[
            # -1300.0,
            # -1100.0,
            # -900.0,
            # -700.0,
            # -400.0,
            # -250.0,
            # -50.0,
            # 50.0,
            # 250.0,
            # 400.0,
            # 700.0,
            # 900.0,
            # 1100.0,
            # 1300.0,
            # ],
           impactMax=3. * u.mm,
           phiBinDeflectionCoverage=3,
        ),  
    
        # SeedingAlgorithmConfigArg(
        #     zBinNeighborsTop = [
        #     [0,0],
        #     [-1,0],
        #     [-1,0],
        #     [-1,0],
        #     [-1,0],
        #     [-1,0],
        #     [-1,1],
        #     [0,1],
        #     [0,1],
        #     [0,1],
        #     [0,1],
        #     [0,1],
        #     [0,0],
        #     ], 
        #     zBinNeighborsBottom = [
        #     [0,1],
        #     [0,1],
        #     [0,1],  
        #     [0,1],
        #     [0,1],
        #     [0,1],
        #     [0,0],
        #     [-1,0],
        #     [-1,0],
        #     [-1,0],
        #     [-1,0],
        #     [-1,0],
        #     [-1,0],
        #     ],
        #     numPhiNeighbors = 2, 
        # ),

        TruthEstimatedSeedingAlgorithmConfigArg(deltaR=(10.0 * u.mm, None)),
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared
        if truthSmearedSeeded
        else SeedingAlgorithm.TruthEstimated
        if truthEstimatedSeeded
        else SeedingAlgorithm.Default,
        geoSelectionConfigFile=geometrySelection,
        outputDirRoot=outputDir,
        rnd=rnd,  # only used by SeedingAlgorithm.TruthSmeared
        logLevel=acts.logging.DEBUG,

        initialVarInflation=[50, 50, 50, 50, 50, 50],


    )

    addCKFTracks(
        s,
        trackingGeometry,
        field,
        TrackSelectorConfig(pt=(ptMin * u.MeV, None), nMeasurementsMin=7),
        outputDirRoot=outputDir,
        outputDirCsv=outputDir / "csv" if outputCsv else None,
        nMeasCutOff = nMeasCutOff,
    )

    addAmbiguityResolution(
       s,
       AmbiguityResolutionConfig(maximumSharedHits=3),
       outputDirRoot=outputDir,
    )

    return s


if "__main__" == __name__:
    options = getArgumentParser().parse_args()

    Inputdir = options.indir
    Outputdir = options.outdir

    #srcdir = Path(__file__).resolve().parent.parent.parent.parent
    srcdir = Path.cwd().parent

    geo_example_dir = Path(options.geo_dir)

    options.outdir.mkdir(exist_ok=True, parents=True)

    assert geo_example_dir.exists(), "Detector example input directory missing"

    detector, trackingGeometry, decorators = buildALICE3Geometry(
        geo_example_dir,
        material=not options.no_material,
    ) # GenericDetector.create()

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    inputParticlePath = Path(Inputdir) / "pythia8_particles.root"
    if not inputParticlePath.exists():
        inputParticlePath = None

    runCKFTracks(
        trackingGeometry,
        decorators,
        field=field,
        geometrySelection=srcdir 
        / "acts/bin/geoSelection-alice3-cfg10.json",
        #srcdir
        #/ "Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json",
        digiConfigFile=srcdir 
        / "acts/bin/alice3-smearing-config.json",
        #srcdir
        #/ "Examples/Algorithms/Digitization/share/default-smearing-config-generic.json",
        outputCsv=True,
        truthSmearedSeeded=False,
        truthEstimatedSeeded=False,
        inputParticlePath=inputParticlePath,
        outputDir=Outputdir,
        NumEvents=options.nEvts,
        EtaMin=options.etaMin,
        EtaMax=options.etaMax,
        MaxSeedsPerSpM=options.sf_maxSeedsPerSpM,
        CotThetaMax=options.sf_cotThetaMax,
        SigmaScattering=options.sf_sigmaScattering,
        RadLengthPerSeed=options.sf_radLengthPerSeed,
        ImpactMax=options.sf_impactMax,
        MaxPtScattering=options.sf_maxPtScattering,
        DeltaRMin=options.sf_deltaRMin,
        DeltaRMax=options.sf_deltaRMax,
    ).run()
