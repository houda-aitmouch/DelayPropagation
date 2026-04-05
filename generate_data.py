"""
AIR-ROBUST — Génération de données fictives
Produit 3 fichiers dans data/:
  1. LEG_FUTUR_011125_301125.xlsx   (vols - format opérationnel RAM)
  2. DOA_PROGRAME_011125_301125.xlsx (équipages - format opérationnel RAM)
  3. reference_params_RAM.csv        (paramètres simulation)

Les lignes sont proportionnelles entre les deux bases:
chaque vol non-annulé a un CA + CC dans DOA_PROGRAME.
"""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# FLOTTE Royal Air Maroc
# ══════════════════════════════════════════════════════════════
FLEET = {
    "CNROH":("73H","AT","000","JY159"),"CNROI":("73H","AT","000","JY160"),
    "CNROJ":("73H","AT","000","JY161"),"CNROK":("73H","AT","000","JY162"),
    "CNROL":("73H","AT","000","JY163"),"CNROM":("73H","AT","000","JY164"),
    "CNROU":("73H","AT","000","JY159"),"CNRGK":("73H","AT","000","JY159"),
    "CNMAX1":("7M8","AT","000","JY170"),"CNMAX2":("7M8","AT","000","JY171"),
    "CNMAX3":("7M8","AT","000","JY172"),"CNMAX4":("7M8","AT","000","JY173"),
    "CNRGB":("788","AT","000","JY180"),"CNRGC":("788","AT","000","JY181"),
    "CNRGD":("788","AT","000","JY182"),
    "CNRGE":("789","AT","000","JY185"),"CNRGF":("789","AT","000","JY186"),
    "CNCOB":("AT7","RXP","001","JY70"),"CNCOC":("AT7","RXP","001","JY71"),
    "CNCOD":("AT7","RXP","001","JY72"),"CNCOE":("AT7","RXP","001","JY73"),
    "CNRGT":("E90","AT","000","JY190"),"CNRGU":("E90","AT","000","JY191"),
}

SUBTYPE_TO_AC   = {"73H":"738","7M8":"7M8","788":"788","789":"789","AT7":"AT7","E90":"E90"}
SUBTYPE_TO_QUAL = {"73H":"B737","7M8":"B737","788":"B737,B787","789":"B737,B787","AT7":"ATR","E90":"B737,B787,E90"}

ROUTES = [
    ("CMN","RAK",409,410,7,30,55),("CMN","RAK",8115,414,7,45,50),
    ("CMN","AGA",417,418,10,0,65),("CMN","AGA",419,420,13,0,65),
    ("CMN","FEZ",421,422,8,0,50),("CMN","FEZ",423,424,10,0,50),
    ("CMN","TNG",425,426,7,0,45),("CMN","TNG",427,428,14,0,45),
    ("CMN","OUD",429,430,9,0,80),("CMN","NDR",431,432,13,0,75),
    ("CMN","EUN",1410,1411,7,30,105),("CMN","EUN",1412,1413,18,15,105),
    ("CMN","VIL",1422,1423,7,25,140),
    ("CMN","CDG",700,701,6,0,175),("CMN","CDG",702,703,15,0,175),
    ("CMN","ORY",704,705,14,0,180),("CMN","MAD",706,707,7,0,130),
    ("CMN","BCN",708,709,13,30,145),("CMN","LYS",790,791,8,5,165),
    ("CMN","MRS",792,793,15,30,160),("CMN","AMS",714,715,8,0,210),
    ("CMN","BRU",716,717,17,30,200),("CMN","FRA",718,719,8,30,220),
    ("CMN","FCO",720,721,9,0,195),("CMN","MXP",722,723,17,30,190),
    ("CMN","LHR",724,725,7,0,195),("CMN","GVA",726,727,16,0,185),
    ("CMN","TLS",728,729,8,0,135),("CMN","NCE",730,731,9,0,170),
    ("CMN","ZRH",732,733,17,30,200),("CMN","AGP",988,989,11,10,75),
    ("CMN","NAP",934,935,15,0,195),
    ("CMN","DKR",500,501,9,0,195),("CMN","ABJ",534,535,14,10,270),
    ("CMN","TUN",570,571,7,50,180),("CMN","ALG",572,573,10,0,120),
    ("CMN","CAI",270,271,12,5,320),("CMN","NKC",521,522,17,0,175),
    ("CMN","OXB",593,594,22,35,165),("CMN","BJL",577,578,22,0,200),
    ("CMN","LOS",540,541,8,0,300),
    ("CMN","JFK",200,201,1,0,480),("CMN","MTL",202,203,2,30,450),
    ("CMN","IAD",204,205,2,0,490),("CMN","DXB",210,211,4,0,390),
    ("CMN","JED",220,221,5,0,330),("CMN","DOH",222,223,3,0,370),
]

DOM_REGS  = ["CNCOB","CNCOC","CNCOD","CNCOE","CNRGT","CNRGU"]
EURO_REGS = ["CNROH","CNROI","CNROJ","CNROK","CNROL","CNROM","CNROU","CNRGK"]
MAX_REGS  = ["CNMAX1","CNMAX2","CNMAX3","CNMAX4"]
WIDE_REGS = ["CNRGB","CNRGC","CNRGD","CNRGE","CNRGF"]
DOM_AP    = {"CMN","RAK","AGA","TNG","FEZ","OUD","NDR","EUN","VIL"}
LONG_DEST = {"JFK","MTL","IAD","DXB","JED","DOH"}
LEG_STATES = ["NEW"]*9 + ["CNL"]

NON_FLT = [("SB4","SBA"),("SB6","SBA"),("CRE","LVE"),("PDO","OFF"),("BUR","ACT"),("SIM","ACT"),("GND","ACT")]
CA_IDS = list(range(45100,45700)); random.shuffle(CA_IDS)
CC_IDS = list(range(8000,9700));   random.shuffle(CC_IDS)
ca_i = cc_i = 0

def pick_reg(dur, orig, dest):
    if dur >= 300: return random.choice(WIDE_REGS)
    if orig in DOM_AP and dest in DOM_AP and dur <= 90: return random.choice(DOM_REGS)
    if dur >= 200: return random.choice(MAX_REGS + EURO_REGS[-2:])
    return random.choice(EURO_REGS + MAX_REGS[:2])

def fday(dt):    return dt.strftime("%Y%m%d")
def fslash(dt):  return dt.strftime("%d/%m/%Y")
def fhm(h,m):   return f"{h:02d}{m:02d}"
def fhmc(h,m):  return f"{h:02d}:{m:02d}"
def blk(d):     return f"{d//60:02d}:{d%60:02d}"

def addm(h,m,mins):
    t=h*60+m+mins; d=t//1440; t=t%1440
    return t//60, t%60, d

def generate_all():
    global ca_i, cc_i
    os.makedirs("data", exist_ok=True)
    leg_rows, doa_rows = [], []
    start, end = datetime(2025,11,1), datetime(2025,11,30)

    print("="*60)
    print("  AIR-ROBUST — Génération données fictives format opérationnel RAM")
    print(f"  Période: {fslash(start)} → {fslash(end)}")
    print("="*60)

    for d in range((end-start).days+1):
        dt = start + timedelta(days=d)
        day8, days = fday(dt), fslash(dt)
        day_routes = [r for r in ROUTES if random.random() < 0.88]

        for orig,dest,fna,fnr,dh,dm,dur in day_routes:
            dm2 = dm+random.randint(-10,15); dh2=dh
            if dm2<0: dm2+=60; dh2-=1
            if dm2>=60: dm2-=60; dh2+=1
            dh2=max(0,min(23,dh2))
            reg=pick_reg(dur,orig,dest); sub,owner,logn,ver=FLEET[reg]
            state=random.choice(LEG_STATES)
            ah,am,dex=addm(dh2,dm2,dur); adt=dt+timedelta(days=dex)

            leg_rows.append({"FN_CARRIER":"AT","FN_NUMBER":str(fna),"FN_SUFFIX":None,
                "JOINT_FN_CARRIER_1":None,"JOINT_FN_CARRIER_2":None,"JOINT_FN_CARRIER_3":None,
                "DAY_OF_ORIGIN":day8,"AC_OWNER":owner,"AC_SUBTYPE":sub,
                "AC_LOGICAL_NO":logn,"AC_VERSION":ver,"AC_PRBD":None,"AC_REGISTRATION":reg,
                "DEP_AP_SCHED":orig,"ARR_AP_SCHED":dest,"DEP_AP_ACTUAL":orig,"ARR_AP_ACTUAL":dest,
                "LEG_STATE":state,"LEG_TYPE":"J",
                "DEP_DAY_SCHED":day8,"DEP_TIME_SCHED":fhm(dh2,dm2),
                "ARR_DAY_SCHED":fday(adt),"ARR_TIME_SCHED":fhm(ah,am)})

            turn=random.randint(120,300) if dest in LONG_DEST else (random.randint(60,120) if dur>=200 else random.randint(40,75))
            rdh,rdm,rd=addm(ah,am,turn); rdt=adt+timedelta(days=rd)
            rah,ram,rd2=addm(rdh,rdm,dur); radt=rdt+timedelta(days=rd2)
            sr=random.choice(LEG_STATES)

            leg_rows.append({"FN_CARRIER":"AT","FN_NUMBER":str(fnr),"FN_SUFFIX":None,
                "JOINT_FN_CARRIER_1":None,"JOINT_FN_CARRIER_2":None,"JOINT_FN_CARRIER_3":None,
                "DAY_OF_ORIGIN":day8,"AC_OWNER":owner,"AC_SUBTYPE":sub,
                "AC_LOGICAL_NO":logn,"AC_VERSION":ver,"AC_PRBD":None,"AC_REGISTRATION":reg,
                "DEP_AP_SCHED":dest,"ARR_AP_SCHED":orig,"DEP_AP_ACTUAL":dest,"ARR_AP_ACTUAL":orig,
                "LEG_STATE":sr,"LEG_TYPE":"J",
                "DEP_DAY_SCHED":fday(rdt),"DEP_TIME_SCHED":fhm(rdh,rdm),
                "ARR_DAY_SCHED":fday(radt),"ARR_TIME_SCHED":fhm(rah,ram)})

            if state != "CNL":
                acc=SUBTYPE_TO_AC[sub]; quals=SUBTYPE_TO_QUAL[sub]
                caid=CA_IDS[ca_i%len(CA_IDS)]; ca_i+=1
                ccid=CC_IDS[cc_i%len(CC_IDS)]; cc_i+=1
                roff=60 if dur<=90 else 120
                for cid,rank,pos in [(caid,"CA","CA"),(ccid,"CC","CC")]:
                    rph,rpm,_=addm(dh2,dm2,-roff)
                    if rph<0: rph+=24
                    doa_rows.append({"CREW_ID":cid,"RANK_":rank,"AC_QUALS":quals,
                        "DATE_UTC":days,"DATE_LT":days,
                        "REPORT_UTC":fhmc(rph,rpm),"REPORT_LT":fhmc((rph+1)%24,rpm),
                        "END_DATE_UTC":fslash(adt),"END_DATE_LT":fslash(adt),
                        "RELEASE_UTC":None,"RELEASE_LT":None,
                        "START_OF_DUTY":"1" if dest!="CMN" else "0",
                        "END_OF_DUTY":"0" if dest!="CMN" else "1",
                        "TAGS":None,"TAG_GROUP":None,"POS":pos,
                        "ACTIVITY":str(fna),"ACTIVITY_GROUP":"FLT",
                        "ORIGINE":orig,"DESTINATION":dest,
                        "START_LT":fhmc((dh2+1)%24,dm2),"START_UTC":fhmc(dh2,dm2),
                        "END_UTC":fhmc(ah,am),"END_LT":fhmc((ah+1)%24,am),
                        "A_C":acc,"LAYOVER":None,"HOTEL":None,
                        "BLOCK_HOURS":blk(dur),"BLC":blk(dur),"AUGMENTATION":None})
                    if sr != "CNL":
                        rrph,rrpm,_=addm(rdh,rdm,-roff)
                        if rrph<0: rrph+=24
                        rlh,rlm,_=addm(rah,ram,30)
                        doa_rows.append({"CREW_ID":cid,"RANK_":rank,"AC_QUALS":quals,
                            "DATE_UTC":days,"DATE_LT":days,
                            "REPORT_UTC":fhmc(rrph,rrpm),"REPORT_LT":fhmc((rrph+1)%24,rrpm),
                            "END_DATE_UTC":fslash(radt),"END_DATE_LT":fslash(radt),
                            "RELEASE_UTC":fhmc(rlh,rlm),"RELEASE_LT":fhmc((rlh+1)%24,rlm),
                            "START_OF_DUTY":"0","END_OF_DUTY":"1",
                            "TAGS":None,"TAG_GROUP":None,"POS":pos,
                            "ACTIVITY":str(fnr),"ACTIVITY_GROUP":"FLT",
                            "ORIGINE":dest,"DESTINATION":orig,
                            "START_LT":fhmc((rdh+1)%24,rdm),"START_UTC":fhmc(rdh,rdm),
                            "END_UTC":fhmc(rah,ram),"END_LT":fhmc((rah+1)%24,ram),
                            "A_C":acc,"LAYOVER":None,"HOTEL":None,
                            "BLOCK_HOURS":blk(dur),"BLC":blk(dur),"AUGMENTATION":None})

        for _ in range(random.randint(10,20)):
            act,grp=random.choice(NON_FLT)
            cid=random.choice(CA_IDS[:100]+CC_IDS[:200])
            rank="CA" if cid>=45000 else "CC"
            quals=random.choice(["B737","B737,B787","B737,B787,E90","ATR"])
            sh=random.randint(0,6) if grp=="SBA" else (0 if grp in ("OFF","LVE") else random.randint(8,10))
            el=None if grp=="LVE" else f"{sh+random.randint(4,8):02d}:00"
            sl=f"{sh:02d}:{'01' if grp=='SBA' else '00'}"
            doa_rows.append({"CREW_ID":cid,"RANK_":rank,"AC_QUALS":quals,
                "DATE_UTC":days,"DATE_LT":days,
                "REPORT_UTC":f"{max(0,sh-1):02d}:{sl.split(':')[1]}",
                "REPORT_LT":sl,
                "END_DATE_UTC":days,"END_DATE_LT":days,
                "RELEASE_UTC":el,"RELEASE_LT":el,
                "START_OF_DUTY":"0","END_OF_DUTY":"0",
                "TAGS":None,"TAG_GROUP":None,"POS":rank if grp!="ACT" else None,
                "ACTIVITY":act,"ACTIVITY_GROUP":grp,
                "ORIGINE":"CMN","DESTINATION":"CMN",
                "START_LT":sl,"START_UTC":f"{max(0,sh-1):02d}:{sl.split(':')[1]}",
                "END_UTC":el,"END_LT":el,
                "A_C":None,"LAYOVER":None,"HOTEL":None,
                "BLOCK_HOURS":"00:00","BLC":blk(random.randint(120,480)),"AUGMENTATION":None})

    leg_cols=["FN_CARRIER","FN_NUMBER","FN_SUFFIX","JOINT_FN_CARRIER_1","JOINT_FN_CARRIER_2",
        "JOINT_FN_CARRIER_3","DAY_OF_ORIGIN","AC_OWNER","AC_SUBTYPE","AC_LOGICAL_NO",
        "AC_VERSION","AC_PRBD","AC_REGISTRATION","DEP_AP_SCHED","ARR_AP_SCHED",
        "DEP_AP_ACTUAL","ARR_AP_ACTUAL","LEG_STATE","LEG_TYPE",
        "DEP_DAY_SCHED","DEP_TIME_SCHED","ARR_DAY_SCHED","ARR_TIME_SCHED"]
    doa_cols=["CREW_ID","RANK_","AC_QUALS","DATE_UTC","DATE_LT","REPORT_UTC","REPORT_LT",
        "END_DATE_UTC","END_DATE_LT","RELEASE_UTC","RELEASE_LT","START_OF_DUTY","END_OF_DUTY",
        "TAGS","TAG_GROUP","POS","ACTIVITY","ACTIVITY_GROUP","ORIGINE","DESTINATION",
        "START_LT","START_UTC","END_UTC","END_LT","A_C","LAYOVER","HOTEL",
        "BLOCK_HOURS","BLC","AUGMENTATION"]

    df_leg=pd.DataFrame(leg_rows)[leg_cols].sort_values(["DAY_OF_ORIGIN","DEP_TIME_SCHED"]).reset_index(drop=True)
    df_doa=pd.DataFrame(doa_rows)[doa_cols].sort_values(["DATE_LT","CREW_ID","START_LT"]).reset_index(drop=True)

    df_leg.to_excel("data/LEG_FUTUR_011125_301125.xlsx",index=False,sheet_name="Export Worksheet")
    df_doa.to_excel("data/DOA_PROGRAME_011125_301125.xlsx",index=False,sheet_name="Export Worksheet")

    params=[]
    for ac,t in {"B737-800":45,"B737-MAX8":45,"B787-8":90,"B787-9":90,"ATR72-600":30,"E190":40}.items():
        params.append({"param_type":"min_turnaround_min","aircraft_type":ac,"value":t,"description":f"Turnaround min {ac}"})
    params+=[{"param_type":"gamma_shape","aircraft_type":"ALL","value":2.0,"description":"alpha Gamma"},
             {"param_type":"gamma_scale","aircraft_type":"ALL","value":15.0,"description":"theta Gamma"}]
    for tr,v in [("Normal->Normal",0.65),("Normal->Alerte",0.28),("Normal->Bloque",0.07),
                 ("Alerte->Normal",0.25),("Alerte->Alerte",0.52),("Alerte->Bloque",0.23),
                 ("Bloque->Normal",0.08),("Bloque->Alerte",0.28),("Bloque->Bloque",0.64)]:
        params.append({"param_type":"markov_transition","aircraft_type":tr,"value":v,"description":f"Markov {tr}"})
    pd.DataFrame(params).to_csv("data/reference_params_RAM.csv",index=False)

    flt=len(df_doa[df_doa["ACTIVITY_GROUP"]=="FLT"])
    cnl=len(df_leg[df_leg["LEG_STATE"]=="CNL"])
    apts=sorted(set(df_leg["DEP_AP_SCHED"])|set(df_leg["ARR_AP_SCHED"]))
    print(f"\n  Fichiers générés:")
    print(f"    data/LEG_FUTUR_011125_301125.xlsx   : {len(df_leg)} vols")
    print(f"    data/DOA_PROGRAME_011125_301125.xlsx : {len(df_doa)} lignes ({flt} FLT)")
    print(f"    data/reference_params_RAM.csv        : {len(params)} params")
    print(f"    Annulés: {cnl} ({cnl/len(df_leg)*100:.1f}%) | Aéroports: {len(apts)}")
    print("="*60)

if __name__ == "__main__":
    generate_all()
