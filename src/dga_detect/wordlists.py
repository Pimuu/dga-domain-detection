"""Embedded vocabularies used by the synthetic data generator.

Nothing is downloaded: every word list lives in this file so the project is
fully self-contained and reproducible offline.

* ``LEGIT_WORDS``      - common English / web words used to build benign,
                         human-chosen domain names (e.g. ``cloudshop``).
* ``DGA_WORDS``        - a *disjoint* vocabulary used only by the held-out
                         "wordlist" DGA family (in the spirit of Suppobox /
                         Matsnu, which concatenate dictionary words).  Keeping
                         it disjoint from ``LEGIT_WORDS`` means the detector
                         cannot simply memorise the benign vocabulary.
* ``LEGIT_PREFIXES`` / ``LEGIT_SUFFIXES`` - affixes common in real brand and
                         service names.
"""

LEGIT_WORDS = """
able account action active adventure air alpha amazing analytics angel animal
answer apple apps art arts atlas audio auto baby bank base basic bay beach bear
beauty best bike bird black blog blue board boat body book books box brain brand
bridge bright build business buy cafe camera camp capital car card care cars
case cash cast cat center chain change chat cheap chef city class clean clear
click clinic cloud club coach code coffee color comfort company connect cook
cool core corner country craft create credit cross crown cube cup daily data
day deal deals design dev diamond digital direct doctor dog door dream drive
earth easy eco edge edu energy engine enter epic event events expert express
eye face fact family farm fashion fast field film finance fire first fish fit
fitness flash flight flow flower fly focus food force forest form forward free
fresh friend fun future game games garden gate gear get gift global go gold
golf good green grid group grow guide hair happy health heart help hero hill
home hope host hotel house hub idea image info inside iron island jet job jobs
joy just key kids king kitchen lab lake land last law learn legal life light
line link lion live local logic love lucky mail main maker map market master
media medical meet metro mind mobile money moon motor mountain move music
nation native nature net network news next night north note nova ocean office
one online open orange order page paint paper park part party pay peak people
perfect pet phone photo pixel place plan planet play plus point power press
prime print pro project pure quick radio rain red rent report rest river road
rock room root royal run safe sale save school science sea search secure sell
service shape share ship shop shopping show signal silver simple site sky smart
snow social soft solar solution sound space spark sport sports spot spring
square star start state station step stone store story street studio style sun
super supply sure swift system talk task team tech test time today tool tools
top tour town toy track trade travel tree trip true trust truth tube union unit
up urban value view village vision visit voice wave way web well west white
wild wind wine wise wolf wood word work world write yard yellow young zone
""".split()

# Disjoint from LEGIT_WORDS (checked by tests/test_data.py).
DGA_WORDS = """
abbey ankle anvil apron arrow attic bacon badge bagel ballot bamboo banjo
barrel basin beacon beetle blanket blossom bonnet bottle bucket buckle butter
cabin cactus candle canyon carpet castle cellar cherry chimney cider cinder
clover cobalt collar copper cotton cradle crayon cricket crystal cushion dagger
daisy denim dinner donkey dragon drawer eagle elbow ember falcon feather fiddle
fossil fountain galaxy garlic ginger glacier goblet granite gravel guitar hammer
harbor harvest helmet hockey honey hornet jacket jasmine jelly jungle kernel
kettle ladder lantern lemon lettuce lizard locket lumber magnet mango marble
meadow mirror mitten muffin napkin nectar needle nickel noodle oyster paddle
pebble pepper pickle pillow pistol pocket pollen puddle pumpkin puppet rabbit
raven ribbon saddle salmon shovel sickle spider spoon squirrel stapler sugar
summit sweater tablet teapot thimble thistle thunder timber tomato trumpet tulip
turtle velvet violin waffle walnut whistle willow window winter wizard zebra
""".split()

LEGIT_PREFIXES = ["my", "get", "the", "go", "i", "e", "we", "try", "use", "best"]
LEGIT_SUFFIXES = ["hub", "ly", "ify", "app", "online", "hq", "now", "pro",
                  "co", "labs", "zone", "box", "io", "ist", "er"]
