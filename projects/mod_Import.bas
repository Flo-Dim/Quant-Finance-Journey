Option Explicit

'===================================================================
' MODULE : mod_Import
'
' BUT : remplacer le copier-coller manuel des extracts banquiers par
' un import piloté par un bouton :
'   1) l'utilisateur choisit les 2 fichiers Excel des banquiers
'   2) le module détecte tout seul lequel est le POS et lequel est
'      le MVT (peu importe l'ordre de sélection)
'   3) il retrouve les colonnes utiles par NOM D'EN-TÊTE (pas par
'      lettre de colonne), donc insensible à un changement d'ordre
'      des colonnes chez un banquier
'   4) il remplit "Report" (ISIN, nb parts, PAM, coupons, encours
'      initial) et archive les données utiles dans une feuille
'      cachée de vérification
'
' Les noms d'en-tête à rechercher NE SONT PAS codés en dur : ils
' sont dans la feuille "Config_Import", modifiable sans toucher au
' code si un banquier change l'intitulé d'une colonne.
'===================================================================

Private Const NOM_FEUILLE_CONFIG As String = "Config_Import"
Private Const NOM_FEUILLE_VERIF As String = "Verif_Import"
Private Const NOM_FEUILLE_REPORT As String = "Report"
Private Const LIGNE_ENTETE As Long = 1
Private Const PREMIERE_LIGNE_REPORT As Long = 31


'###################################################################
' 1) POINT D'ENTREE - à assigner au bouton
'###################################################################

Sub ImporterFichiersBanquiers()

    Dim cheminsFichiers As Variant
    Dim wbBanquier1 As Workbook, wbBanquier2 As Workbook
    Dim wsBrut1 As Worksheet, wsBrut2 As Worksheet
    Dim wsMVT As Worksheet, wsPOS As Worksheet
    Dim mapMVT As Object, mapPOS As Object
    Dim typeFichier1 As String, typeFichier2 As String

    Application.ScreenUpdating = False
    On Error GoTo GestionErreur

    S'assurerQueLaConfigExiste

    ' --- 1. Sélection des 2 fichiers ---
    cheminsFichiers = Application.GetOpenFilename( _
        FileFilter:="Fichiers Excel (*.xlsx;*.xls),*.xlsx;*.xls", _
        Title:="Sélectionner les 2 fichiers banquiers (POS + MVT)", _
        MultiSelect:=True)

    If Not IsArray(cheminsFichiers) Then
        MsgBox "Import annulé : aucun fichier sélectionné.", vbInformation
        GoTo Fin
    End If

    If UBound(cheminsFichiers) - LBound(cheminsFichiers) + 1 <> 2 Then
        MsgBox "Merci de sélectionner exactement 2 fichiers (un POS et un MVT).", vbExclamation
        GoTo Fin
    End If

    ' --- 2. Ouverture en lecture seule ---
    Set wbBanquier1 = Workbooks.Open(cheminsFichiers(LBound(cheminsFichiers)), UpdateLinks:=0, ReadOnly:=True)
    Set wbBanquier2 = Workbooks.Open(cheminsFichiers(LBound(cheminsFichiers) + 1), UpdateLinks:=0, ReadOnly:=True)
    Set wsBrut1 = wbBanquier1.Worksheets(1)
    Set wsBrut2 = wbBanquier2.Worksheets(1)

    ' --- 3. Détection automatique POS / MVT ---
    typeFichier1 = DetecterTypeFichier(wsBrut1)
    typeFichier2 = DetecterTypeFichier(wsBrut2)

    If typeFichier1 = "INCONNU" Or typeFichier2 = "INCONNU" Then
        MsgBox "Impossible de reconnaître le type d'un des deux fichiers." & vbCrLf & _
               "Vérifie que les en-têtes attendus (voir feuille """ & NOM_FEUILLE_CONFIG & """) sont bien présents.", vbCritical
        GoTo Fin
    End If

    If typeFichier1 = typeFichier2 Then
        MsgBox "Les 2 fichiers sélectionnés semblent être du même type (" & typeFichier1 & ")." & vbCrLf & _
               "Merci de vérifier qu'il s'agit bien d'un fichier POS et d'un fichier MVT.", vbExclamation
        GoTo Fin
    End If

    If typeFichier1 = "POS" Then
        Set wsPOS = wsBrut1
        Set wsMVT = wsBrut2
    Else
        Set wsPOS = wsBrut2
        Set wsMVT = wsBrut1
    End If

    ' --- 4. Index des colonnes par nom d'en-tête ---
    Set mapMVT = ConstruireIndexColonnes(wsMVT)
    Set mapPOS = ConstruireIndexColonnes(wsPOS)

    If Not ColonnesRequisesPresentes(mapMVT, "MVT") Then GoTo Fin
    If Not ColonnesRequisesPresentes(mapPOS, "POS") Then GoTo Fin

    ' --- 5. Traitement + écriture dans Report + archivage ---
    TraiterExtractMVT wsMVT, mapMVT
    TraiterExtractPOS wsPOS, mapPOS

    MsgBox "Import terminé avec succès.", vbInformation

Fin:
    On Error Resume Next
    If Not wbBanquier1 Is Nothing Then wbBanquier1.Close SaveChanges:=False
    If Not wbBanquier2 Is Nothing Then wbBanquier2.Close SaveChanges:=False
    Application.ScreenUpdating = True
    Exit Sub

GestionErreur:
    MsgBox "Erreur pendant l'import : " & Err.Description, vbCritical
    Resume Fin

End Sub


'###################################################################
' 2) DETECTION DU TYPE DE FICHIER ET INDEXATION DES COLONNES
'###################################################################

' Détecte si une feuille brute est un extract POS ou MVT,
' en cherchant les en-têtes distinctifs définis dans Config_Import.
Function DetecterTypeFichier(ws As Worksheet) As String
    Dim derniereCol As Long, c As Long
    Dim entete As String
    Dim motCleMVT As String, motClePOS As String

    motCleMVT = ValeurParametre("MOT_CLE_MVT")   ' ex : "Code Mvt"
    motClePOS = ValeurParametre("MOT_CLE_POS")   ' ex : "PAM"

    derniereCol = ws.Cells(LIGNE_ENTETE, ws.Columns.Count).End(xlToLeft).Column

    For c = 1 To derniereCol
        entete = Trim(ws.Cells(LIGNE_ENTETE, c).Value)
        If StrComp(entete, motClePOS, vbTextCompare) = 0 Then
            DetecterTypeFichier = "POS"
            Exit Function
        ElseIf StrComp(entete, motCleMVT, vbTextCompare) = 0 Then
            DetecterTypeFichier = "MVT"
            Exit Function
        End If
    Next c

    DetecterTypeFichier = "INCONNU"
End Function

' Construit un dictionnaire {NOM D'ENTETE EN MAJUSCULE -> N° colonne}
Function ConstruireIndexColonnes(ws As Worksheet) As Object
    Dim dict As Object
    Dim derniereCol As Long, c As Long
    Dim nomEntete As String

    Set dict = CreateObject("Scripting.Dictionary")
    derniereCol = ws.Cells(LIGNE_ENTETE, ws.Columns.Count).End(xlToLeft).Column

    For c = 1 To derniereCol
        nomEntete = Trim(ws.Cells(LIGNE_ENTETE, c).Value)
        If nomEntete <> "" Then
            If Not dict.Exists(UCase(nomEntete)) Then
                dict.Add UCase(nomEntete), c
            End If
        End If
    Next c

    Set ConstruireIndexColonnes = dict
End Function

' Renvoie le n° de colonne correspondant à un champ technique
' (ex: "ISIN", "PAM"...) pour un type d'extract donné ("MVT"/"POS"),
' en passant par la feuille Config_Import.
Function ColonneChamp(map As Object, typeExtract As String, champTechnique As String) As Long
    Dim nomEntete As String
    nomEntete = NomEnteteConfig(typeExtract, champTechnique)

    If nomEntete = "" Then
        ColonneChamp = 0
    ElseIf map.Exists(UCase(nomEntete)) Then
        ColonneChamp = map(UCase(nomEntete))
    Else
        ColonneChamp = 0
    End If
End Function

' Vérifie que toutes les colonnes attendues (définies dans
' Config_Import) sont bien trouvées ; affiche un message clair sinon.
Function ColonnesRequisesPresentes(map As Object, typeExtract As String) As Boolean
    Dim wsConfig As Worksheet
    Dim derniereLigne As Long, i As Long
    Dim champTechnique As String, nomEntete As String
    Dim manquants As String

    Set wsConfig = ThisWorkbook.Worksheets(NOM_FEUILLE_CONFIG)
    derniereLigne = wsConfig.Cells(wsConfig.Rows.Count, "A").End(xlUp).Row
    manquants = ""

    For i = 2 To derniereLigne
        If UCase(Trim(wsConfig.Cells(i, "A").Value)) = UCase(typeExtract) Then
            champTechnique = Trim(wsConfig.Cells(i, "B").Value)
            nomEntete = Trim(wsConfig.Cells(i, "C").Value)
            If Not map.Exists(UCase(nomEntete)) Then
                manquants = manquants & "  - " & champTechnique & " (entête recherché : """ & nomEntete & """)" & vbCrLf
            End If
        End If
    Next i

    If manquants <> "" Then
        MsgBox "Colonnes introuvables dans l'extract " & typeExtract & " :" & vbCrLf & manquants & _
               vbCrLf & "Vérifie les noms d'en-tête dans la feuille """ & NOM_FEUILLE_CONFIG & """.", vbExclamation
        ColonnesRequisesPresentes = False
    Else
        ColonnesRequisesPresentes = True
    End If
End Function


'###################################################################
' 3) TRAITEMENT DE L'EXTRACT MVT (coupons + encours initial)
'###################################################################

Sub TraiterExtractMVT(wsMVT As Worksheet, mapMVT As Object)

    Dim colCodeMvt As Long, colIsin As Long, colMontant As Long, colSens As Long
    Dim derniereLigne As Long, i As Long
    Dim codeMvt As String, isin As String
    Dim montant As Double
    Dim dictCoupons As Object
    Dim wsReport As Worksheet, wsVerif As Worksheet
    Dim motifCoupon As String, prefixeApport As String, sensCredit As String
    Dim encoursInitial As Double

    colCodeMvt = ColonneChamp(mapMVT, "MVT", "CODE_MVT")
    colIsin = ColonneChamp(mapMVT, "MVT", "ISIN")
    colMontant = ColonneChamp(mapMVT, "MVT", "MONTANT")
    colSens = ColonneChamp(mapMVT, "MVT", "SENS")

    motifCoupon = ValeurParametre("MOTIF_COUPON")          ' ex : "LK02:*Intérêts*"
    prefixeApport = ValeurParametre("PREFIXE_APPORT")      ' ex : "CZ53"
    sensCredit = ValeurParametre("SENS_CREDIT")            ' ex : "C:Credit"

    Set dictCoupons = CreateObject("Scripting.Dictionary")
    Set wsVerif = ObtenirFeuilleVerif()

    derniereLigne = wsMVT.Cells(wsMVT.Rows.Count, colCodeMvt).End(xlUp).Row
    encoursInitial = 0

    For i = LIGNE_ENTETE + 1 To derniereLigne
        codeMvt = Trim(wsMVT.Cells(i, colCodeMvt).Value)
        isin = Trim(wsMVT.Cells(i, colIsin).Value)
        montant = 0
        If IsNumeric(wsMVT.Cells(i, colMontant).Value) Then montant = wsMVT.Cells(i, colMontant).Value

        ' --- Coupons / intérêts ---
        If codeMvt Like motifCoupon Then
            If Not dictCoupons.Exists(isin) Then dictCoupons.Add isin, 0
            dictCoupons(isin) = dictCoupons(isin) + montant
            ArchiverVerif wsVerif, "MVT-Coupon", isin, montant, codeMvt
        End If

        ' --- Encours initial (apports crédités) ---
        If Left(codeMvt, Len(prefixeApport)) = prefixeApport _
           And StrComp(Trim(wsMVT.Cells(i, colSens).Value), sensCredit, vbTextCompare) = 0 Then
            encoursInitial = encoursInitial + montant
        End If
    Next i

    Set wsReport = ThisWorkbook.Worksheets(NOM_FEUILLE_REPORT)
    EcrireCouponsDansReport wsReport, dictCoupons
    wsReport.Range("R12").Value = encoursInitial

End Sub

' Reporte les totaux de coupons en face de chaque ISIN déjà présent
' dans Report (colonne K), à partir de la ligne 31.
Sub EcrireCouponsDansReport(wsReport As Worksheet, dictCoupons As Object)
    Dim derniereLigneReport As Long, i As Long
    Dim isinReport As String

    derniereLigneReport = wsReport.Cells(wsReport.Rows.Count, "B").End(xlUp).Row

    For i = PREMIERE_LIGNE_REPORT To derniereLigneReport
        isinReport = Trim(wsReport.Cells(i, "B").Value)
        If isinReport <> "" And dictCoupons.Exists(isinReport) Then
            wsReport.Cells(i, "K").Value = dictCoupons(isinReport)
        End If
    Next i
End Sub


'###################################################################
' 4) TRAITEMENT DE L'EXTRACT POS (ISIN, nb parts, PAM + cash)
'###################################################################

Sub TraiterExtractPOS(wsPOS As Worksheet, mapPOS As Object)

    Dim colIsin As Long, colNbParts As Long, colPAM As Long, colValorisation As Long
    Dim derniereLigne As Long, i As Long, j As Long
    Dim isin As String, isinMonetaire As String
    Dim wsReport As Worksheet, wsVerif As Worksheet
    Dim ligneReport As Long
    Dim lignesProduits As Collection
    Dim ligneProduit(1 To 3) As Variant
    Dim valorisationCashTotal As Double
    Dim derniereLigneReportAvant As Long

    colIsin = ColonneChamp(mapPOS, "POS", "ISIN")
    colNbParts = ColonneChamp(mapPOS, "POS", "NB_PARTS")
    colPAM = ColonneChamp(mapPOS, "POS", "PAM")
    colValorisation = ColonneChamp(mapPOS, "POS", "VALORISATION")
    isinMonetaire = ValeurParametre("ISIN_MONETAIRE")   ' ex : "DE000A2QBG39"

    Set lignesProduits = New Collection
    Set wsVerif = ObtenirFeuilleVerif()
    valorisationCashTotal = 0

    derniereLigne = wsPOS.Cells(wsPOS.Rows.Count, colIsin).End(xlUp).Row

    ' --- TODO À VALIDER ENSEMBLE ---
    ' Ici, une ligne est considérée "cash" (mise à la fin, agrégée)
    ' si son ISIN est vide OU correspond à l'ISIN monétaire de
    ' référence défini dans Config_Import. Si le cash "pur" apparaît
    ' différemment dans le vrai fichier POS, on ajustera ce test.
    For i = LIGNE_ENTETE + 1 To derniereLigne
        isin = Trim(wsPOS.Cells(i, colIsin).Value)

        If isin = "" Or StrComp(isin, isinMonetaire, vbTextCompare) = 0 Then
            If IsNumeric(wsPOS.Cells(i, colValorisation).Value) Then
                valorisationCashTotal = valorisationCashTotal + wsPOS.Cells(i, colValorisation).Value
            End If
            ArchiverVerif wsVerif, "POS-Cash", isin, wsPOS.Cells(i, colValorisation).Value, "agrégé en CASH"
        Else
            ligneProduit(1) = isin
            ligneProduit(2) = wsPOS.Cells(i, colNbParts).Value
            ligneProduit(3) = wsPOS.Cells(i, colPAM).Value
            lignesProduits.Add Array(ligneProduit(1), ligneProduit(2), ligneProduit(3))
            ArchiverVerif wsVerif, "POS-Produit", isin, ligneProduit(2), "PAM=" & ligneProduit(3)
        End If
    Next i

    ' --- Écriture dans Report, à partir de la ligne 31 ---
    Set wsReport = ThisWorkbook.Worksheets(NOM_FEUILLE_REPORT)
    derniereLigneReportAvant = wsReport.Cells(wsReport.Rows.Count, "B").End(xlUp).Row
    ligneReport = PREMIERE_LIGNE_REPORT

    For j = 1 To lignesProduits.Count
        wsReport.Cells(ligneReport, "B").Value = lignesProduits(j)(0)
        wsReport.Cells(ligneReport, "E").Value = lignesProduits(j)(1)
        wsReport.Cells(ligneReport, "F").Value = lignesProduits(j)(2)
        ligneReport = ligneReport + 1
    Next j

    ' --- Ligne CASH, toujours en dernière position ---
    wsReport.Cells(ligneReport, "B").Value = "CASH"
    wsReport.Cells(ligneReport, "E").Value = valorisationCashTotal / 1000
    wsReport.Cells(ligneReport, "F").Value = 100
    ligneReport = ligneReport + 1

    ' --- Nettoyage des lignes en trop d'un import précédent ---
    If derniereLigneReportAvant >= ligneReport Then
        wsReport.Rows(ligneReport & ":" & derniereLigneReportAvant).ClearContents
    End If

End Sub


'###################################################################
' 5) FEUILLE DE VERIFICATION (archivage des données utiles)
'###################################################################

Function ObtenirFeuilleVerif() As Worksheet
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(NOM_FEUILLE_VERIF)
    On Error GoTo 0

    If ws Is Nothing Then
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))
        ws.Name = NOM_FEUILLE_VERIF
        ws.Range("A1:E1").Value = Array("Horodatage", "Type", "ISIN", "Valeur", "Détail")
        ws.Visible = xlSheetVeryHidden
    End If

    Set ObtenirFeuilleVerif = ws
End Function

Sub ArchiverVerif(wsVerif As Worksheet, typeLigne As String, isin As String, valeur As Variant, detail As String)
    Dim ligne As Long
    ligne = wsVerif.Cells(wsVerif.Rows.Count, "A").End(xlUp).Row + 1

    wsVerif.Cells(ligne, "A").Value = Now
    wsVerif.Cells(ligne, "B").Value = typeLigne
    wsVerif.Cells(ligne, "C").Value = isin
    wsVerif.Cells(ligne, "D").Value = valeur
    wsVerif.Cells(ligne, "E").Value = detail
End Sub


'###################################################################
' 6) FEUILLE DE CONFIGURATION (modifiable sans toucher au code)
'###################################################################

' Lit le nom d'en-tête à rechercher pour un champ technique donné.
Function NomEnteteConfig(typeExtract As String, champTechnique As String) As String
    Dim wsConfig As Worksheet
    Dim derniereLigne As Long, i As Long

    Set wsConfig = ThisWorkbook.Worksheets(NOM_FEUILLE_CONFIG)
    derniereLigne = wsConfig.Cells(wsConfig.Rows.Count, "A").End(xlUp).Row

    For i = 2 To derniereLigne
        If UCase(Trim(wsConfig.Cells(i, "A").Value)) = UCase(typeExtract) _
           And UCase(Trim(wsConfig.Cells(i, "B").Value)) = UCase(champTechnique) Then
            NomEnteteConfig = Trim(wsConfig.Cells(i, "C").Value)
            Exit Function
        End If
    Next i

    NomEnteteConfig = ""
End Function

' Lit un paramètre simple (mots-clés de détection, ISIN monétaire...)
' dans le petit tableau "Paramètres" de la feuille Config_Import.
Function ValeurParametre(nomParametre As String) As String
    Dim wsConfig As Worksheet
    Dim derniereLigne As Long, i As Long

    Set wsConfig = ThisWorkbook.Worksheets(NOM_FEUILLE_CONFIG)
    derniereLigne = wsConfig.Cells(wsConfig.Rows.Count, "E").End(xlUp).Row

    For i = 2 To derniereLigne
        If UCase(Trim(wsConfig.Cells(i, "E").Value)) = UCase(nomParametre) Then
            ValeurParametre = Trim(wsConfig.Cells(i, "F").Value)
            Exit Function
        End If
    Next i

    ValeurParametre = ""
End Function

' Crée la feuille Config_Import avec des valeurs par défaut si elle
' n'existe pas déjà (basées sur les en-têtes réels observés).
' À lancer une seule fois (ou laisser s'exécuter à chaque import,
' elle ne touche à rien si la feuille existe déjà).
Sub S'assurerQueLaConfigExiste()
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(NOM_FEUILLE_CONFIG)
    On Error GoTo 0

    If Not ws Is Nothing Then Exit Sub

    Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))
    ws.Name = NOM_FEUILLE_CONFIG

    ' --- Tableau 1 : mapping des colonnes ---
    ws.Range("A1:C1").Value = Array("Type extract", "Champ technique", "Nom entête à rechercher")
    ws.Range("A1:C1").Font.Bold = True

    Dim mapping As Variant
    mapping = Array( _
        Array("MVT", "CODE_MVT", "Code Mvt"), _
        Array("MVT", "ISIN", "Isin"), _
        Array("MVT", "MONTANT", "Net Euro"), _
        Array("MVT", "SENS", "Sens"), _
        Array("POS", "ISIN", "ISIN"), _
        Array("POS", "NB_PARTS", "Qte Titres"), _
        Array("POS", "PAM", "PAM"), _
        Array("POS", "VALORISATION", "Valorisation") _
    )

    Dim i As Long
    For i = LBound(mapping) To UBound(mapping)
        ws.Cells(i + 2, "A").Value = mapping(i)(0)
        ws.Cells(i + 2, "B").Value = mapping(i)(1)
        ws.Cells(i + 2, "C").Value = mapping(i)(2)
    Next i

    ' --- Tableau 2 : paramètres divers ---
    ws.Range("E1:F1").Value = Array("Paramètre", "Valeur")
    ws.Range("E1:F1").Font.Bold = True

    Dim parametres As Variant
    parametres = Array( _
        Array("MOT_CLE_MVT", "Code Mvt"), _
        Array("MOT_CLE_POS", "PAM"), _
        Array("MOTIF_COUPON", "LK02:*Intérêts*"), _
        Array("PREFIXE_APPORT", "CZ53"), _
        Array("SENS_CREDIT", "C:Credit"), _
        Array("ISIN_MONETAIRE", "DE000A2QBG39") _
    )

    For i = LBound(parametres) To UBound(parametres)
        ws.Cells(i + 2, "E").Value = parametres(i)(0)
        ws.Cells(i + 2, "F").Value = parametres(i)(1)
    Next i

    ws.Columns("A:F").AutoFit
End Sub


'###################################################################
' 7) OUTIL : créer le bouton dans la feuille active (à lancer 1 fois)
'###################################################################

Sub CreerBoutonImport()
    Dim btn As Button
    Set btn = ActiveSheet.Buttons.Add(ActiveSheet.Range("B2").Left, ActiveSheet.Range("B2").Top, 220, 40)
    btn.OnAction = "ImporterFichiersBanquiers"
    btn.Caption = "Importer les fichiers banquiers"
End Sub
