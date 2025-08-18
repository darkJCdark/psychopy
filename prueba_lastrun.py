#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Agosto 17, 2025, at 18:10
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'prueba'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\joseg\\OneDrive\\Documentos\\PROYECTO SEMINARIO\\prueba_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instruccion" ---
    text = visual.TextStim(win=win, name='text',
        text='Bienvenid@ a la prueba\n\nEn este experimento, verás en cada ensayo 4 pares de letras.\nA veces los pares formarán PALABRAS REALES de 2 letras (por ejemplo: WE, TO, AS). Otras veces, los pares serán COMBINACIONES ALEATORIAS de letras. Cada letra estará escrita en NEGRITA o CURSIVA.\n\nTu tarea es la siguiente: \nDespués de ver las letras, te preguntaremos por una letra específica.\nDebes indicar:\n1. Qué letra era.\n2. En qué formato estaba (negrita o cursiva).\n\nEn la pantalla de respuesta verás 4 opciones que combinan letra y formato.\nDebes seleccionar la opción correcta.\n\nTen en cuenta esto: \na. Presta atención tanto a la letra como a su formato.\nb. No hay límite de tiempo, pero responde lo más rápido y preciso posible.\nc. Usa el ratón para seleccionar la opción.\n\nPresiona ESPACIO para comenzar.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=1.4, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "letras" ---
    letra1 = visual.TextStim(win=win, name='letra1',
        text='',
        font='Arial',
        pos=(-0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    letra2 = visual.TextStim(win=win, name='letra2',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    letra3 = visual.TextStim(win=win, name='letra3',
        text='',
        font='Arial',
        pos=(-0.025, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    letra4 = visual.TextStim(win=win, name='letra4',
        text='',
        font='Arial',
        pos=(0.025, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    letra5 = visual.TextStim(win=win, name='letra5',
        text='',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    letra6 = visual.TextStim(win=win, name='letra6',
        text='',
        font='Arial',
        pos=(0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    letra7 = visual.TextStim(win=win, name='letra7',
        text='',
        font='Arial',
        pos=(-0.025, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    letra8 = visual.TextStim(win=win, name='letra8',
        text='',
        font='Arial',
        pos=(0.025, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    
    # --- Initialize components for Routine "espera" ---
    Espera = visual.TextStim(win=win, name='Espera',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "pRespuesta" ---
    Alternativa_1 = visual.TextStim(win=win, name='Alternativa_1',
        text='',
        font='Arial',
        pos=(-0.3, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Alternativa_2 = visual.TextStim(win=win, name='Alternativa_2',
        text='',
        font='Arial',
        pos=(-0.1, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Alternativa_3 = visual.TextStim(win=win, name='Alternativa_3',
        text='',
        font='Arial',
        pos=(0.1, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Alternativa_4 = visual.TextStim(win=win, name='Alternativa_4',
        text='',
        font='Arial',
        pos=(0.3, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "Despedida" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Has llegado al final de la prueba.\n¡Muchas gracias por tu participación! :D\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Instruccion" ---
    # create an object to store info about Routine Instruccion
    Instruccion = data.Routine(
        name='Instruccion',
        components=[text, key_resp],
    )
    Instruccion.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Instruccion
    Instruccion.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruccion.tStart = globalClock.getTime(format='float')
    Instruccion.status = STARTED
    thisExp.addData('Instruccion.started', Instruccion.tStart)
    Instruccion.maxDuration = None
    # keep track of which components have finished
    InstruccionComponents = Instruccion.components
    for thisComponent in Instruccion.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruccion" ---
    Instruccion.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Instruccion.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruccion.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruccion" ---
    for thisComponent in Instruccion.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruccion
    Instruccion.tStop = globalClock.getTime(format='float')
    Instruccion.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instruccion.stopped', Instruccion.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Instruccion" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=5.0, 
        method='fullRandom', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('config.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "letras" ---
        # create an object to store info about Routine letras
        letras = data.Routine(
            name='letras',
            components=[letra1, letra2, letra3, letra4, letra5, letra6, letra7, letra8],
        )
        letras.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        letra1.setText(letter1)
        letra1.setFont(type_letter1)
        letra2.setText(letter2)
        letra2.setFont(type_letter2)
        letra3.setText(letter3)
        letra3.setFont(type_letter3)
        letra4.setText(letter4)
        letra4.setFont(type_letter4)
        letra5.setText(letter5)
        letra5.setFont(type_letter5)
        letra6.setText(letter6)
        letra6.setFont(type_letter6)
        letra7.setText(letter7)
        letra7.setFont(type_letter7)
        letra8.setText(letter8)
        letra8.setFont(type_letter8)
        # store start times for letras
        letras.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        letras.tStart = globalClock.getTime(format='float')
        letras.status = STARTED
        thisExp.addData('letras.started', letras.tStart)
        letras.maxDuration = None
        # keep track of which components have finished
        letrasComponents = letras.components
        for thisComponent in letras.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "letras" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        letras.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *letra1* updates
            
            # if letra1 is starting this frame...
            if letra1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                letra1.frameNStart = frameN  # exact frame index
                letra1.tStart = t  # local t and not account for scr refresh
                letra1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letra1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letra1.started')
                # update status
                letra1.status = STARTED
                letra1.setAutoDraw(True)
            
            # if letra1 is active this frame...
            if letra1.status == STARTED:
                # update params
                pass
            
            # if letra1 is stopping this frame...
            if letra1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letra1.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    letra1.tStop = t  # not accounting for scr refresh
                    letra1.tStopRefresh = tThisFlipGlobal  # on global time
                    letra1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letra1.stopped')
                    # update status
                    letra1.status = FINISHED
                    letra1.setAutoDraw(False)
            
            # *letra2* updates
            
            # if letra2 is starting this frame...
            if letra2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                letra2.frameNStart = frameN  # exact frame index
                letra2.tStart = t  # local t and not account for scr refresh
                letra2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letra2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letra2.started')
                # update status
                letra2.status = STARTED
                letra2.setAutoDraw(True)
            
            # if letra2 is active this frame...
            if letra2.status == STARTED:
                # update params
                pass
            
            # if letra2 is stopping this frame...
            if letra2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letra2.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    letra2.tStop = t  # not accounting for scr refresh
                    letra2.tStopRefresh = tThisFlipGlobal  # on global time
                    letra2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letra2.stopped')
                    # update status
                    letra2.status = FINISHED
                    letra2.setAutoDraw(False)
            
            # *letra3* updates
            
            # if letra3 is starting this frame...
            if letra3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                letra3.frameNStart = frameN  # exact frame index
                letra3.tStart = t  # local t and not account for scr refresh
                letra3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letra3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letra3.started')
                # update status
                letra3.status = STARTED
                letra3.setAutoDraw(True)
            
            # if letra3 is active this frame...
            if letra3.status == STARTED:
                # update params
                pass
            
            # if letra3 is stopping this frame...
            if letra3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letra3.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    letra3.tStop = t  # not accounting for scr refresh
                    letra3.tStopRefresh = tThisFlipGlobal  # on global time
                    letra3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letra3.stopped')
                    # update status
                    letra3.status = FINISHED
                    letra3.setAutoDraw(False)
            
            # *letra4* updates
            
            # if letra4 is starting this frame...
            if letra4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                letra4.frameNStart = frameN  # exact frame index
                letra4.tStart = t  # local t and not account for scr refresh
                letra4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letra4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letra4.started')
                # update status
                letra4.status = STARTED
                letra4.setAutoDraw(True)
            
            # if letra4 is active this frame...
            if letra4.status == STARTED:
                # update params
                pass
            
            # if letra4 is stopping this frame...
            if letra4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letra4.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    letra4.tStop = t  # not accounting for scr refresh
                    letra4.tStopRefresh = tThisFlipGlobal  # on global time
                    letra4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letra4.stopped')
                    # update status
                    letra4.status = FINISHED
                    letra4.setAutoDraw(False)
            
            # *letra5* updates
            
            # if letra5 is starting this frame...
            if letra5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                letra5.frameNStart = frameN  # exact frame index
                letra5.tStart = t  # local t and not account for scr refresh
                letra5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letra5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letra5.started')
                # update status
                letra5.status = STARTED
                letra5.setAutoDraw(True)
            
            # if letra5 is active this frame...
            if letra5.status == STARTED:
                # update params
                pass
            
            # if letra5 is stopping this frame...
            if letra5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letra5.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    letra5.tStop = t  # not accounting for scr refresh
                    letra5.tStopRefresh = tThisFlipGlobal  # on global time
                    letra5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letra5.stopped')
                    # update status
                    letra5.status = FINISHED
                    letra5.setAutoDraw(False)
            
            # *letra6* updates
            
            # if letra6 is starting this frame...
            if letra6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                letra6.frameNStart = frameN  # exact frame index
                letra6.tStart = t  # local t and not account for scr refresh
                letra6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letra6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letra6.started')
                # update status
                letra6.status = STARTED
                letra6.setAutoDraw(True)
            
            # if letra6 is active this frame...
            if letra6.status == STARTED:
                # update params
                pass
            
            # if letra6 is stopping this frame...
            if letra6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letra6.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    letra6.tStop = t  # not accounting for scr refresh
                    letra6.tStopRefresh = tThisFlipGlobal  # on global time
                    letra6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letra6.stopped')
                    # update status
                    letra6.status = FINISHED
                    letra6.setAutoDraw(False)
            
            # *letra7* updates
            
            # if letra7 is starting this frame...
            if letra7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                letra7.frameNStart = frameN  # exact frame index
                letra7.tStart = t  # local t and not account for scr refresh
                letra7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letra7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letra7.started')
                # update status
                letra7.status = STARTED
                letra7.setAutoDraw(True)
            
            # if letra7 is active this frame...
            if letra7.status == STARTED:
                # update params
                pass
            
            # if letra7 is stopping this frame...
            if letra7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letra7.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    letra7.tStop = t  # not accounting for scr refresh
                    letra7.tStopRefresh = tThisFlipGlobal  # on global time
                    letra7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letra7.stopped')
                    # update status
                    letra7.status = FINISHED
                    letra7.setAutoDraw(False)
            
            # *letra8* updates
            
            # if letra8 is starting this frame...
            if letra8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                letra8.frameNStart = frameN  # exact frame index
                letra8.tStart = t  # local t and not account for scr refresh
                letra8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letra8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letra8.started')
                # update status
                letra8.status = STARTED
                letra8.setAutoDraw(True)
            
            # if letra8 is active this frame...
            if letra8.status == STARTED:
                # update params
                pass
            
            # if letra8 is stopping this frame...
            if letra8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letra8.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    letra8.tStop = t  # not accounting for scr refresh
                    letra8.tStopRefresh = tThisFlipGlobal  # on global time
                    letra8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letra8.stopped')
                    # update status
                    letra8.status = FINISHED
                    letra8.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                letras.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in letras.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "letras" ---
        for thisComponent in letras.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for letras
        letras.tStop = globalClock.getTime(format='float')
        letras.tStopRefresh = tThisFlipGlobal
        thisExp.addData('letras.stopped', letras.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if letras.maxDurationReached:
            routineTimer.addTime(-letras.maxDuration)
        elif letras.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "espera" ---
        # create an object to store info about Routine espera
        espera = data.Routine(
            name='espera',
            components=[Espera],
        )
        espera.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for espera
        espera.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        espera.tStart = globalClock.getTime(format='float')
        espera.status = STARTED
        thisExp.addData('espera.started', espera.tStart)
        espera.maxDuration = None
        # keep track of which components have finished
        esperaComponents = espera.components
        for thisComponent in espera.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "espera" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        espera.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.7:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Espera* updates
            
            # if Espera is starting this frame...
            if Espera.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Espera.frameNStart = frameN  # exact frame index
                Espera.tStart = t  # local t and not account for scr refresh
                Espera.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Espera, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Espera.started')
                # update status
                Espera.status = STARTED
                Espera.setAutoDraw(True)
            
            # if Espera is active this frame...
            if Espera.status == STARTED:
                # update params
                pass
            
            # if Espera is stopping this frame...
            if Espera.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Espera.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    Espera.tStop = t  # not accounting for scr refresh
                    Espera.tStopRefresh = tThisFlipGlobal  # on global time
                    Espera.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Espera.stopped')
                    # update status
                    Espera.status = FINISHED
                    Espera.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                espera.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in espera.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "espera" ---
        for thisComponent in espera.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for espera
        espera.tStop = globalClock.getTime(format='float')
        espera.tStopRefresh = tThisFlipGlobal
        thisExp.addData('espera.stopped', espera.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if espera.maxDurationReached:
            routineTimer.addTime(-espera.maxDuration)
        elif espera.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.700000)
        
        # --- Prepare to start Routine "pRespuesta" ---
        # create an object to store info about Routine pRespuesta
        pRespuesta = data.Routine(
            name='pRespuesta',
            components=[Alternativa_1, Alternativa_2, Alternativa_3, Alternativa_4, mouse],
        )
        pRespuesta.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Alternativa_1.setText(letterFoil1)
        Alternativa_1.setFont(typeFoil1)
        Alternativa_2.setText(letterFoil2)
        Alternativa_2.setFont(typeFoil2)
        Alternativa_3.setText(letterFoil3)
        Alternativa_3.setFont(typeFoil3)
        Alternativa_4.setText(letterFoil4)
        Alternativa_4.setFont(typeFoil4)
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from codeMouse
        win.mouseVisible = True
        
        Alternativa_1.value = 1
        Alternativa_2.value = 2
        Alternativa_3.value = 3
        Alternativa_4.value = 4
        correcto = 0   # reiniciamos en cada trial
        respuesta = None
        
        # Run 'Begin Routine' code from codeVisual
        # Begin Routine
        starPos = int(thisTrial['starPosition']) - 1
        star_position_coords = [
            (-0.25, 0), (-0.2, 0),
            (-0.025, 0.2), (0.025, 0.2),
            (0.2, 0), (0.25, 0),
            (-0.025, -0.2), (0.025, -0.2)
        ]
        
        # crear la cruz con '-' excepto en la posición marcada
        crossStims = []
        for i, pos in enumerate(star_position_coords):
            if i == starPos:
                text = '*'
            else:
                text = '-'
            crossStims.append(visual.TextStim(win=win, text=text, pos=pos, height=0.05))
        
        # store start times for pRespuesta
        pRespuesta.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pRespuesta.tStart = globalClock.getTime(format='float')
        pRespuesta.status = STARTED
        thisExp.addData('pRespuesta.started', pRespuesta.tStart)
        pRespuesta.maxDuration = None
        # keep track of which components have finished
        pRespuestaComponents = pRespuesta.components
        for thisComponent in pRespuesta.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pRespuesta" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        pRespuesta.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Alternativa_1* updates
            
            # if Alternativa_1 is starting this frame...
            if Alternativa_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Alternativa_1.frameNStart = frameN  # exact frame index
                Alternativa_1.tStart = t  # local t and not account for scr refresh
                Alternativa_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Alternativa_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Alternativa_1.started')
                # update status
                Alternativa_1.status = STARTED
                Alternativa_1.setAutoDraw(True)
            
            # if Alternativa_1 is active this frame...
            if Alternativa_1.status == STARTED:
                # update params
                pass
            
            # *Alternativa_2* updates
            
            # if Alternativa_2 is starting this frame...
            if Alternativa_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Alternativa_2.frameNStart = frameN  # exact frame index
                Alternativa_2.tStart = t  # local t and not account for scr refresh
                Alternativa_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Alternativa_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Alternativa_2.started')
                # update status
                Alternativa_2.status = STARTED
                Alternativa_2.setAutoDraw(True)
            
            # if Alternativa_2 is active this frame...
            if Alternativa_2.status == STARTED:
                # update params
                pass
            
            # *Alternativa_3* updates
            
            # if Alternativa_3 is starting this frame...
            if Alternativa_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Alternativa_3.frameNStart = frameN  # exact frame index
                Alternativa_3.tStart = t  # local t and not account for scr refresh
                Alternativa_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Alternativa_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Alternativa_3.started')
                # update status
                Alternativa_3.status = STARTED
                Alternativa_3.setAutoDraw(True)
            
            # if Alternativa_3 is active this frame...
            if Alternativa_3.status == STARTED:
                # update params
                pass
            
            # *Alternativa_4* updates
            
            # if Alternativa_4 is starting this frame...
            if Alternativa_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Alternativa_4.frameNStart = frameN  # exact frame index
                Alternativa_4.tStart = t  # local t and not account for scr refresh
                Alternativa_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Alternativa_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Alternativa_4.started')
                # update status
                Alternativa_4.status = STARTED
                Alternativa_4.setAutoDraw(True)
            
            # if Alternativa_4 is active this frame...
            if Alternativa_4.status == STARTED:
                # update params
                pass
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse.started', t)
                # update status
                mouse.status = STARTED
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([Alternativa_1, Alternativa_2, Alternativa_3, Alternativa_4], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse):
                                gotValidClick = True
                                mouse.clicked_name.append(obj.name)
                        if not gotValidClick:
                            mouse.clicked_name.append(None)
                        x, y = mouse.getPos()
                        mouse.x.append(x)
                        mouse.y.append(y)
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from codeMouse
            for obj in [Alternativa_1, Alternativa_2, Alternativa_3, Alternativa_4]:
                if mouse.isPressedIn(obj):
                    respuesta_elegida = obj.value
                    if respuesta_elegida == correctAnswer:
                        correcto = 1
                    else:
                        correcto = 0
            # Run 'Each Frame' code from codeVisual
            for stim in crossStims:
                stim.draw()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pRespuesta.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pRespuesta.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pRespuesta" ---
        for thisComponent in pRespuesta.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pRespuesta
        pRespuesta.tStop = globalClock.getTime(format='float')
        pRespuesta.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pRespuesta.stopped', pRespuesta.tStop)
        # store data for trials (TrialHandler)
        trials.addData('mouse.x', mouse.x)
        trials.addData('mouse.y', mouse.y)
        trials.addData('mouse.leftButton', mouse.leftButton)
        trials.addData('mouse.midButton', mouse.midButton)
        trials.addData('mouse.rightButton', mouse.rightButton)
        trials.addData('mouse.time', mouse.time)
        trials.addData('mouse.clicked_name', mouse.clicked_name)
        # Run 'End Routine' code from codeMouse
        thisExp.addData('respuesta', respuesta_elegida)
        thisExp.addData('correctAnswer', correctAnswer)
        thisExp.addData('correcto', correcto)
        # the Routine "pRespuesta" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Despedida" ---
    # create an object to store info about Routine Despedida
    Despedida = data.Routine(
        name='Despedida',
        components=[text_2],
    )
    Despedida.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Despedida
    Despedida.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Despedida.tStart = globalClock.getTime(format='float')
    Despedida.status = STARTED
    thisExp.addData('Despedida.started', Despedida.tStart)
    Despedida.maxDuration = None
    # keep track of which components have finished
    DespedidaComponents = Despedida.components
    for thisComponent in Despedida.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Despedida" ---
    Despedida.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Despedida.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Despedida.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Despedida" ---
    for thisComponent in Despedida.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Despedida
    Despedida.tStop = globalClock.getTime(format='float')
    Despedida.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Despedida.stopped', Despedida.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Despedida.maxDurationReached:
        routineTimer.addTime(-Despedida.maxDuration)
    elif Despedida.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
