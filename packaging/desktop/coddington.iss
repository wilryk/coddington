; Inno Setup script for the Coddington Windows installer.
;
; Builds a single Coddington-Setup-<version>.exe from the PyInstaller output
; in dist\Coddington. Run coddington.spec first -- this script does not
; invoke PyInstaller itself; scripts\build_release.ps1 does both in order.
;
; Build (from the repository root):
;
;     iscc packaging\desktop\coddington.iss /DMyAppVersion=0.1.0
;
; MyAppVersion defaults to 0.0.0 if not passed on the command line, so a
; direct "iscc coddington.iss" still produces something runnable while
; developing the script.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#define MyAppName "Coddington"
#define MyAppPublisher "Ryker Optics"
#define MyAppExeName "Coddington.exe"
#define MyAppURL "https://github.com/wilryk/coddington"

[Setup]
; Fixed once and never changed -- Inno uses this to recognise "the same
; app" across versions, which is what makes upgrade/uninstall work later.
AppId={{F219423D-FEED-4671-8B56-9A1CF06F63F1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
; {autopf}\Coddington under an administrative install, or a per-user
; equivalent under a non-administrative one -- see PrivilegesRequired below.
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
; Two installs of the same unsigned build in a row would otherwise ask a
; confusing "which directory" question the second time.
DisableDirPage=no
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
; No admin rights required: installs to the user's own profile if the
; installer isn't run elevated, to Program Files if it is. Either way, no
; UAC prompt is forced on top of the unsigned-exe SmartScreen warning.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist\installer
OutputBaseFilename=Coddington-Setup-{#MyAppVersion}
SetupIconFile=coddington.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; This is the unsigned first release: no code-signing certificate exists
; yet, so Windows SmartScreen will warn on both this installer and the app
; it installs. That is expected -- see the README for what a user sees.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; Everything PyInstaller collected, folder structure intact.
Source: "..\..\dist\Coddington\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Offered, not forced -- someone who just wanted the shortcuts installed
; shouldn't have a browser window pop open on them.
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} now"; Flags: nowait postinstall skipifsilent unchecked
