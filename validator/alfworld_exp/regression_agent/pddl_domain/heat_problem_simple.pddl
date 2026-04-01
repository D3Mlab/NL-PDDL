
(define (problem plan_5306)
    (:domain put_task)
    (:objects
        TeddyBear - object
        WineBottle - object
        Lettuce - object
        PepperShaker - object
        Bathtub - object
        Poster - object
        HousePlant - object
        Boots - object
        LightSwitch - object
        CellPhone - object
        RemoteControl - object
        Footstool - object
        SaltShaker - object
        DishSponge - object
        Cup - object
        Curtains - object
        Fork - object
        Vase - object
        Ladle - object
        BasketBall - object
        ScrubBrush - object
        Pillow - object
        Spoon - object
        Newspaper - object
        ToiletPaper - object
        ButterKnife - object
        SoapBar - object
        HandTowel - object
        Book - object
        TennisRacket - object
        SoapBottle - object
        ShowerGlass - object
        BaseballBat - object
        CD - object
        Plunger - object
        Kettle - object
        Window - object
        KeyChain - object
        Pen - object
        Pot - object
        FloorLamp - object
        Spatula - object
        Blinds - object
        LaundryHamperLid - object
        PaperTowel - object
        Glassbottle - object
        Statue - object
        Towel - object
        Tomato - object
        Bowl - object
        Bread - object
        TissueBox - object
        Mug - object
        Cloth - object
        DeskLamp - object
        Chair - object
        Television - object
        Apple - object
        SprayBottle - object
        CreditCard - object
        Pan - object
        Plate - object
        AlarmClock - object
        Knife - object
        Box - object
        Potato - object
        Egg - object
        ToiletPaperRoll - object
        WateringCan - object
        Candle - object
        PaperTowelRoll - object
        StoveKnob - object
        Mirror - object
        Laptop - object
        ShowerDoor - object
        Painting - object
        Pencil - object
        Watch - object
        TeddyBearType - otype
        WineBottleType - otype
        LettuceType - otype
        PepperShakerType - otype
        BathtubType - otype
        PosterType - otype
        HousePlantType - otype
        BootsType - otype
        LightSwitchType - otype
        CellPhoneType - otype
        RemoteControlType - otype
        FootstoolType - otype
        SaltShakerType - otype
        DishSpongeType - otype
        CupType - otype
        CurtainsType - otype
        ForkType - otype
        VaseType - otype
        LadleType - otype
        BasketBallType - otype
        ScrubBrushType - otype
        PillowType - otype
        SpoonType - otype
        NewspaperType - otype
        ToiletPaperType - otype
        ButterKnifeType - otype
        SoapBarType - otype
        HandTowelType - otype
        BookType - otype
        TennisRacketType - otype
        SoapBottleType - otype
        ShowerGlassType - otype
        BaseballBatType - otype
        CDType - otype
        PlungerType - otype
        KettleType - otype
        WindowType - otype
        KeyChainType - otype
        PenType - otype
        PotType - otype
        FloorLampType - otype
        SpatulaType - otype
        BlindsType - otype
        LaundryHamperLidType - otype
        PaperTowelType - otype
        GlassbottleType - otype
        StatueType - otype
        TowelType - otype
        TomatoType - otype
        BowlType - otype
        BreadType - otype
        TissueBoxType - otype
        MugType - otype
        ClothType - otype
        DeskLampType - otype
        ChairType - otype
        TelevisionType - otype
        AppleType - otype
        SprayBottleType - otype
        CreditCardType - otype
        PanType - otype
        PlateType - otype
        AlarmClockType - otype
        KnifeType - otype
        BoxType - otype
        PotatoType - otype
        EggType - otype
        ToiletPaperRollType - otype
        WateringCanType - otype
        CandleType - otype
        PaperTowelRollType - otype
        StoveKnobType - otype
        MirrorType - otype
        LaptopType - otype
        ShowerDoorType - otype
        PaintingType - otype
        PencilType - otype
        WatchType - otype
        CartType - rtype
        StoveBurnerType - rtype
        MicrowaveType - rtype
        LaundryHamperType - rtype
        DiningTableType - rtype
        DrawerType - rtype
        PaintingHangerType - rtype
        BedType - rtype
        OttomanType - rtype
        SafeType - rtype
        GarbageCanType - rtype
        FridgeType - rtype
        ToasterType - rtype
        CoffeeMachineType - rtype
        CoffeeTableType - rtype
        ShelfType - rtype
        TowelHolderType - rtype
        ToiletPaperHangerType - rtype
        DeskType - rtype
        SideTableType - rtype
        ToiletType - rtype
        HandTowelHolderType - rtype
        CabinetType - rtype
        CounterTopType - rtype
        ArmChairType - rtype
        TVStandType - rtype
        BathtubBasinType - rtype
        DresserType - rtype
        SofaType - rtype


        Bowl_bar__plus_00_dot_97_bar__plus_00_dot_91_bar__minus_02_dot_64 - object
        Cup_bar__plus_00_dot_08_bar__plus_01_dot_11_bar__minus_00_dot_76 - object
        Egg_bar__plus_02_dot_06_bar__plus_00_dot_60_bar__minus_02_dot_56 - object
        Egg_bar__minus_01_dot_44_bar__plus_00_dot_94_bar__minus_02_dot_41 - object
        Mug_bar__plus_00_dot_32_bar__plus_00_dot_91_bar__minus_02_dot_71 - object
        Pan_bar__minus_00_dot_47_bar__plus_00_dot_95_bar__minus_02_dot_37 - object
        Plate_bar__plus_01_dot_75_bar__plus_00_dot_56_bar__minus_02_dot_56 - object
        Pot_bar__minus_01_dot_77_bar__plus_00_dot_91_bar__minus_00_dot_70 - object
        Cabinet_bar__plus_00_dot_68_bar__plus_00_dot_50_bar__minus_02_dot_20 - receptacle
        Cabinet_bar__plus_00_dot_68_bar__plus_02_dot_02_bar__minus_02_dot_46 - receptacle
        Cabinet_bar__plus_00_dot_72_bar__plus_02_dot_02_bar__minus_02_dot_46 - receptacle
        Cabinet_bar__minus_00_dot_73_bar__plus_02_dot_02_bar__minus_02_dot_46 - receptacle
        Cabinet_bar__minus_01_dot_18_bar__plus_00_dot_50_bar__minus_02_dot_20 - receptacle
        Cabinet_bar__minus_01_dot_55_bar__plus_00_dot_50_bar__plus_00_dot_38 - receptacle
        Cabinet_bar__minus_01_dot_55_bar__plus_00_dot_50_bar__minus_01_dot_97 - receptacle
        Cabinet_bar__minus_01_dot_69_bar__plus_02_dot_02_bar__minus_02_dot_46 - receptacle
        Cabinet_bar__minus_01_dot_85_bar__plus_02_dot_02_bar__plus_00_dot_38 - receptacle
        CoffeeMachine_bar__minus_01_dot_98_bar__plus_00_dot_90_bar__minus_00_dot_19 - receptacle
        CounterTop_bar__plus_00_dot_69_bar__plus_00_dot_95_bar__minus_02_dot_48 - receptacle
        CounterTop_bar__minus_00_dot_08_bar__plus_01_dot_15_bar_00_dot_00 - receptacle
        CounterTop_bar__minus_01_dot_87_bar__plus_00_dot_95_bar__minus_01_dot_21 - receptacle
        Drawer_bar__plus_00_dot_95_bar__plus_00_dot_22_bar__minus_02_dot_20 - receptacle
        Drawer_bar__plus_00_dot_95_bar__plus_00_dot_39_bar__minus_02_dot_20 - receptacle
        Drawer_bar__plus_00_dot_95_bar__plus_00_dot_56_bar__minus_02_dot_20 - receptacle
        Drawer_bar__plus_00_dot_95_bar__plus_00_dot_71_bar__minus_02_dot_20 - receptacle
        Drawer_bar__plus_00_dot_95_bar__plus_00_dot_83_bar__minus_02_dot_20 - receptacle
        Drawer_bar__minus_01_dot_56_bar__plus_00_dot_33_bar__minus_00_dot_20 - receptacle
        Drawer_bar__minus_01_dot_56_bar__plus_00_dot_66_bar__minus_00_dot_20 - receptacle
        Drawer_bar__minus_01_dot_56_bar__plus_00_dot_84_bar__plus_00_dot_20 - receptacle
        Drawer_bar__minus_01_dot_56_bar__plus_00_dot_84_bar__minus_00_dot_20 - receptacle
        Fridge_bar__minus_02_dot_10_bar__plus_00_dot_00_bar__plus_01_dot_07 - receptacle
        GarbageCan_bar__minus_01_dot_94_bar_00_dot_00_bar__plus_02_dot_03 - receptacle
        Microwave_bar__minus_00_dot_24_bar__plus_01_dot_69_bar__minus_02_dot_53 - receptacle
        Shelf_bar__plus_01_dot_75_bar__plus_00_dot_17_bar__minus_02_dot_56 - receptacle
        Shelf_bar__plus_01_dot_75_bar__plus_00_dot_55_bar__minus_02_dot_56 - receptacle
        Shelf_bar__plus_01_dot_75_bar__plus_00_dot_88_bar__minus_02_dot_56 - receptacle
        StoveBurner_bar__minus_00_dot_04_bar__plus_00_dot_92_bar__minus_02_dot_37 - receptacle
        StoveBurner_bar__minus_00_dot_04_bar__plus_00_dot_92_bar__minus_02_dot_58 - receptacle
        StoveBurner_bar__minus_00_dot_47_bar__plus_00_dot_92_bar__minus_02_dot_37 - receptacle
        StoveBurner_bar__minus_00_dot_47_bar__plus_00_dot_92_bar__minus_02_dot_58 - receptacle
        Toaster_bar__minus_01_dot_84_bar__plus_00_dot_90_bar__plus_00_dot_13 - receptacle
        loc_bar__minus_3_bar__minus_6_bar_3_bar_60 - location
        loc_bar__minus_5_bar_2_bar_3_bar__minus_30 - location
        loc_bar__minus_5_bar__minus_3_bar_3_bar_60 - location
        loc_bar__minus_5_bar__minus_5_bar_2_bar_45 - location
        loc_bar__minus_5_bar__minus_7_bar_3_bar_45 - location
        loc_bar_6_bar__minus_4_bar_2_bar_45 - location
        loc_bar__minus_4_bar_2_bar_3_bar_60 - location
        loc_bar__minus_5_bar__minus_7_bar_2_bar__minus_30 - location
        loc_bar__minus_1_bar__minus_7_bar_2_bar_0 - location
        loc_bar_4_bar__minus_7_bar_2_bar__minus_30 - location
        loc_bar__minus_4_bar_3_bar_2_bar_60 - location
        loc_bar__minus_4_bar_4_bar_2_bar_45 - location
        loc_bar_1_bar__minus_7_bar_1_bar_45 - location
        loc_bar__minus_5_bar_1_bar_3_bar_45 - location
        loc_bar__minus_4_bar__minus_7_bar_2_bar__minus_30 - location
        loc_bar_3_bar__minus_7_bar_2_bar_45 - location
        loc_bar_0_bar__minus_7_bar_2_bar_45 - location
        loc_bar__minus_2_bar__minus_7_bar_2_bar_45 - location
        loc_bar_7_bar__minus_8_bar_2_bar_60 - location
        loc_bar__minus_5_bar__minus_1_bar_3_bar_45 - location
        loc_bar_3_bar_0_bar_3_bar_30 - location
        loc_bar__minus_4_bar__minus_4_bar_0_bar_45 - location
        loc_bar__minus_5_bar__minus_1_bar_3_bar_60 - location
        loc_bar_0_bar__minus_5_bar_2_bar_45 - location
        loc_bar__minus_5_bar__minus_7_bar_2_bar_45 - location
        loc_bar_4_bar__minus_7_bar_2_bar_45 - location
        loc_bar_7_bar__minus_7_bar_3_bar_45 - location
        loc_bar_0_bar__minus_5_bar_0_bar_45 - location
        loc_bar_2_bar__minus_7_bar_2_bar__minus_30 - location
        loc_bar_1_bar__minus_7_bar_2_bar_45 - location
        loc_bar__minus_5_bar__minus_6_bar_3_bar_60 - location
        loc_bar_2_bar__minus_5_bar_2_bar_45 - location
        loc_bar__minus_5_bar_7_bar_3_bar_60 - location
        loc_bar__minus_4_bar_4_bar_3_bar_60 - location
        loc_bar_4_bar_5_bar_3_bar_30 - location
        )


    (:init
        


        (receptacleType Drawer_bar__plus_00_dot_95_bar__plus_00_dot_22_bar__minus_02_dot_20 DrawerType)
        (receptacleType Drawer_bar__minus_01_dot_56_bar__plus_00_dot_33_bar__minus_00_dot_20 DrawerType)
        (receptacleType Shelf_bar__plus_01_dot_75_bar__plus_00_dot_88_bar__minus_02_dot_56 ShelfType)
        (receptacleType Shelf_bar__plus_01_dot_75_bar__plus_00_dot_17_bar__minus_02_dot_56 ShelfType)
        (receptacleType Drawer_bar__plus_00_dot_95_bar__plus_00_dot_39_bar__minus_02_dot_20 DrawerType)
        (receptacleType Cabinet_bar__minus_01_dot_69_bar__plus_02_dot_02_bar__minus_02_dot_46 CabinetType)
        (receptacleType CoffeeMachine_bar__minus_01_dot_98_bar__plus_00_dot_90_bar__minus_00_dot_19 CoffeeMachineType)
        (receptacleType Cabinet_bar__plus_00_dot_72_bar__plus_02_dot_02_bar__minus_02_dot_46 CabinetType)
        (receptacleType StoveBurner_bar__minus_00_dot_04_bar__plus_00_dot_92_bar__minus_02_dot_37 StoveBurnerType)
        (receptacleType Drawer_bar__minus_01_dot_56_bar__plus_00_dot_84_bar__minus_00_dot_20 DrawerType)
        (receptacleType StoveBurner_bar__minus_00_dot_04_bar__plus_00_dot_92_bar__minus_02_dot_58 StoveBurnerType)
        (receptacleType StoveBurner_bar__minus_00_dot_47_bar__plus_00_dot_92_bar__minus_02_dot_37 StoveBurnerType)
        (receptacleType Cabinet_bar__minus_01_dot_18_bar__plus_00_dot_50_bar__minus_02_dot_20 CabinetType)
        (receptacleType Cabinet_bar__plus_00_dot_68_bar__plus_00_dot_50_bar__minus_02_dot_20 CabinetType)
        (receptacleType Cabinet_bar__minus_01_dot_55_bar__plus_00_dot_50_bar__plus_00_dot_38 CabinetType)
        (receptacleType Cabinet_bar__minus_00_dot_73_bar__plus_02_dot_02_bar__minus_02_dot_46 CabinetType)
        (receptacleType CounterTop_bar__minus_01_dot_87_bar__plus_00_dot_95_bar__minus_01_dot_21 CounterTopType)
        (receptacleType CounterTop_bar__minus_00_dot_08_bar__plus_01_dot_15_bar_00_dot_00 CounterTopType)
        (receptacleType Drawer_bar__minus_01_dot_56_bar__plus_00_dot_84_bar__plus_00_dot_20 DrawerType)
        (receptacleType Drawer_bar__plus_00_dot_95_bar__plus_00_dot_56_bar__minus_02_dot_20 DrawerType)
        (receptacleType Shelf_bar__plus_01_dot_75_bar__plus_00_dot_55_bar__minus_02_dot_56 ShelfType)
        (receptacleType Drawer_bar__plus_00_dot_95_bar__plus_00_dot_83_bar__minus_02_dot_20 DrawerType)
        (receptacleType Cabinet_bar__minus_01_dot_55_bar__plus_00_dot_50_bar__minus_01_dot_97 CabinetType)
        (receptacleType StoveBurner_bar__minus_00_dot_47_bar__plus_00_dot_92_bar__minus_02_dot_58 StoveBurnerType)
        (receptacleType Cabinet_bar__plus_00_dot_68_bar__plus_02_dot_02_bar__minus_02_dot_46 CabinetType)
        (receptacleType GarbageCan_bar__minus_01_dot_94_bar_00_dot_00_bar__plus_02_dot_03 GarbageCanType)
        (receptacleType CounterTop_bar__plus_00_dot_69_bar__plus_00_dot_95_bar__minus_02_dot_48 CounterTopType)
        (receptacleType Microwave_bar__minus_00_dot_24_bar__plus_01_dot_69_bar__minus_02_dot_53 MicrowaveType)
        (receptacleType Toaster_bar__minus_01_dot_84_bar__plus_00_dot_90_bar__plus_00_dot_13 ToasterType)
        (receptacleType Drawer_bar__plus_00_dot_95_bar__plus_00_dot_71_bar__minus_02_dot_20 DrawerType)
        (objectType Plate_bar__plus_01_dot_75_bar__plus_00_dot_56_bar__minus_02_dot_56 PlateType)
        (objectType Mug_bar__plus_00_dot_32_bar__plus_00_dot_91_bar__minus_02_dot_71 MugType)
        (objectType Egg_bar__minus_01_dot_44_bar__plus_00_dot_94_bar__minus_02_dot_41 EggType)
        (objectType Pot_bar__minus_01_dot_77_bar__plus_00_dot_91_bar__minus_00_dot_70 PotType)
        (objectType Pan_bar__minus_00_dot_47_bar__plus_00_dot_95_bar__minus_02_dot_37 PanType)
        (objectType Egg_bar__plus_02_dot_06_bar__plus_00_dot_60_bar__minus_02_dot_56 EggType)
        (objectType Cup_bar__plus_00_dot_08_bar__plus_01_dot_11_bar__minus_00_dot_76 CupType)
        (objectType Bowl_bar__plus_00_dot_97_bar__plus_00_dot_91_bar__minus_02_dot_64 BowlType)
        (isReceptacleObject Plate_bar__plus_01_dot_75_bar__plus_00_dot_56_bar__minus_02_dot_56)
        (isReceptacleObject Mug_bar__plus_00_dot_32_bar__plus_00_dot_91_bar__minus_02_dot_71)
        (isReceptacleObject Pot_bar__minus_01_dot_77_bar__plus_00_dot_91_bar__minus_00_dot_70)
        (isReceptacleObject Pan_bar__minus_00_dot_47_bar__plus_00_dot_95_bar__minus_02_dot_37)
        (isReceptacleObject Cup_bar__plus_00_dot_08_bar__plus_01_dot_11_bar__minus_00_dot_76)
        (isReceptacleObject Bowl_bar__plus_00_dot_97_bar__plus_00_dot_91_bar__minus_02_dot_64)
        (openable Drawer_bar__plus_00_dot_95_bar__plus_00_dot_22_bar__minus_02_dot_20)
        (openable Drawer_bar__minus_01_dot_56_bar__plus_00_dot_33_bar__minus_00_dot_20)
        (openable Drawer_bar__plus_00_dot_95_bar__plus_00_dot_39_bar__minus_02_dot_20)
        (openable Cabinet_bar__minus_01_dot_69_bar__plus_02_dot_02_bar__minus_02_dot_46)
        (openable Cabinet_bar__plus_00_dot_72_bar__plus_02_dot_02_bar__minus_02_dot_46)
        (openable Drawer_bar__minus_01_dot_56_bar__plus_00_dot_84_bar__minus_00_dot_20)
        (openable Cabinet_bar__minus_01_dot_18_bar__plus_00_dot_50_bar__minus_02_dot_20)
        (openable Cabinet_bar__plus_00_dot_68_bar__plus_00_dot_50_bar__minus_02_dot_20)
        (openable Cabinet_bar__minus_01_dot_55_bar__plus_00_dot_50_bar__plus_00_dot_38)
        (openable Fridge_bar__minus_02_dot_10_bar__plus_00_dot_00_bar__plus_01_dot_07)
        (openable Cabinet_bar__minus_00_dot_73_bar__plus_02_dot_02_bar__minus_02_dot_46)
        (openable Drawer_bar__minus_01_dot_56_bar__plus_00_dot_84_bar__plus_00_dot_20)
        (openable Cabinet_bar__minus_01_dot_85_bar__plus_02_dot_02_bar__plus_00_dot_38)
        (openable Drawer_bar__plus_00_dot_95_bar__plus_00_dot_56_bar__minus_02_dot_20)
        (openable Drawer_bar__minus_01_dot_56_bar__plus_00_dot_66_bar__minus_00_dot_20)
        (openable Drawer_bar__plus_00_dot_95_bar__plus_00_dot_83_bar__minus_02_dot_20)
        (openable Cabinet_bar__minus_01_dot_55_bar__plus_00_dot_50_bar__minus_01_dot_97)
        (openable Cabinet_bar__plus_00_dot_68_bar__plus_02_dot_02_bar__minus_02_dot_46)
        (openable Microwave_bar__minus_00_dot_24_bar__plus_01_dot_69_bar__minus_02_dot_53)
        (openable Drawer_bar__plus_00_dot_95_bar__plus_00_dot_71_bar__minus_02_dot_20)
        
        (atLocation agent1 loc_bar_4_bar_5_bar_3_bar_30)
        

        
        (heatable Plate_bar__plus_01_dot_75_bar__plus_00_dot_56_bar__minus_02_dot_56)
        (heatable Mug_bar__plus_00_dot_32_bar__plus_00_dot_91_bar__minus_02_dot_71)
        (heatable Egg_bar__minus_01_dot_44_bar__plus_00_dot_94_bar__minus_02_dot_41)
        (heatable Egg_bar__plus_02_dot_06_bar__plus_00_dot_60_bar__minus_02_dot_56)
        (heatable Cup_bar__plus_00_dot_08_bar__plus_01_dot_11_bar__minus_00_dot_76)    

        
        (inReceptacle Egg_bar__minus_01_dot_44_bar__plus_00_dot_94_bar__minus_02_dot_41 CounterTop_bar__minus_01_dot_87_bar__plus_00_dot_95_bar__minus_01_dot_21)
        (inReceptacle Egg_bar__plus_02_dot_06_bar__plus_00_dot_60_bar__minus_02_dot_56 Shelf_bar__plus_01_dot_75_bar__plus_00_dot_55_bar__minus_02_dot_56)
        (wasInReceptacle  Egg_bar__minus_01_dot_44_bar__plus_00_dot_94_bar__minus_02_dot_41 CounterTop_bar__minus_01_dot_87_bar__plus_00_dot_95_bar__minus_01_dot_21)
        (wasInReceptacle  Egg_bar__plus_02_dot_06_bar__plus_00_dot_60_bar__minus_02_dot_56 Shelf_bar__plus_01_dot_75_bar__plus_00_dot_55_bar__minus_02_dot_56)
        (receptacleAtLocation Cabinet_bar__plus_00_dot_68_bar__plus_00_dot_50_bar__minus_02_dot_20 loc_bar_0_bar__minus_5_bar_2_bar_45)
        (receptacleAtLocation Cabinet_bar__plus_00_dot_68_bar__plus_02_dot_02_bar__minus_02_dot_46 loc_bar_2_bar__minus_7_bar_2_bar__minus_30)
        (receptacleAtLocation Cabinet_bar__plus_00_dot_72_bar__plus_02_dot_02_bar__minus_02_dot_46 loc_bar_4_bar__minus_7_bar_2_bar__minus_30)
        (receptacleAtLocation Cabinet_bar__minus_00_dot_73_bar__plus_02_dot_02_bar__minus_02_dot_46 loc_bar__minus_4_bar__minus_7_bar_2_bar__minus_30)
        (receptacleAtLocation Cabinet_bar__minus_01_dot_18_bar__plus_00_dot_50_bar__minus_02_dot_20 loc_bar__minus_5_bar__minus_5_bar_2_bar_45)
        (receptacleAtLocation Cabinet_bar__minus_01_dot_55_bar__plus_00_dot_50_bar__plus_00_dot_38 loc_bar__minus_4_bar_2_bar_3_bar_60)
        (receptacleAtLocation Cabinet_bar__minus_01_dot_55_bar__plus_00_dot_50_bar__minus_01_dot_97 loc_bar__minus_3_bar__minus_6_bar_3_bar_60)
        (receptacleAtLocation Cabinet_bar__minus_01_dot_69_bar__plus_02_dot_02_bar__minus_02_dot_46 loc_bar__minus_5_bar__minus_7_bar_2_bar__minus_30)
        (receptacleAtLocation Cabinet_bar__minus_01_dot_85_bar__plus_02_dot_02_bar__plus_00_dot_38 loc_bar__minus_5_bar_2_bar_3_bar__minus_30)
        (receptacleAtLocation CoffeeMachine_bar__minus_01_dot_98_bar__plus_00_dot_90_bar__minus_00_dot_19 loc_bar__minus_5_bar__minus_1_bar_3_bar_45)
        (receptacleAtLocation CounterTop_bar__plus_00_dot_69_bar__plus_00_dot_95_bar__minus_02_dot_48 loc_bar_3_bar__minus_7_bar_2_bar_45)
        (receptacleAtLocation CounterTop_bar__minus_00_dot_08_bar__plus_01_dot_15_bar_00_dot_00 loc_bar_3_bar_0_bar_3_bar_30)
        (receptacleAtLocation CounterTop_bar__minus_01_dot_87_bar__plus_00_dot_95_bar__minus_01_dot_21 loc_bar__minus_5_bar__minus_7_bar_3_bar_45)
        (receptacleAtLocation Drawer_bar__plus_00_dot_95_bar__plus_00_dot_22_bar__minus_02_dot_20 loc_bar_6_bar__minus_4_bar_2_bar_45)
        (receptacleAtLocation Drawer_bar__plus_00_dot_95_bar__plus_00_dot_39_bar__minus_02_dot_20 loc_bar_6_bar__minus_4_bar_2_bar_45)
        (receptacleAtLocation Drawer_bar__plus_00_dot_95_bar__plus_00_dot_56_bar__minus_02_dot_20 loc_bar_2_bar__minus_5_bar_2_bar_45)
        (receptacleAtLocation Drawer_bar__plus_00_dot_95_bar__plus_00_dot_71_bar__minus_02_dot_20 loc_bar_7_bar__minus_7_bar_3_bar_45)
        (receptacleAtLocation Drawer_bar__plus_00_dot_95_bar__plus_00_dot_83_bar__minus_02_dot_20 loc_bar_1_bar__minus_7_bar_1_bar_45)
        (receptacleAtLocation Drawer_bar__minus_01_dot_56_bar__plus_00_dot_33_bar__minus_00_dot_20 loc_bar__minus_4_bar_3_bar_2_bar_60)
        (receptacleAtLocation Drawer_bar__minus_01_dot_56_bar__plus_00_dot_66_bar__minus_00_dot_20 loc_bar__minus_5_bar__minus_1_bar_3_bar_60)
        (receptacleAtLocation Drawer_bar__minus_01_dot_56_bar__plus_00_dot_84_bar__plus_00_dot_20 loc_bar__minus_4_bar_4_bar_2_bar_45)
        (receptacleAtLocation Drawer_bar__minus_01_dot_56_bar__plus_00_dot_84_bar__minus_00_dot_20 loc_bar__minus_4_bar__minus_4_bar_0_bar_45)
        (receptacleAtLocation Fridge_bar__minus_02_dot_10_bar__plus_00_dot_00_bar__plus_01_dot_07 loc_bar__minus_4_bar_4_bar_3_bar_60)
        (receptacleAtLocation GarbageCan_bar__minus_01_dot_94_bar_00_dot_00_bar__plus_02_dot_03 loc_bar__minus_5_bar_7_bar_3_bar_60)
        (receptacleAtLocation Microwave_bar__minus_00_dot_24_bar__plus_01_dot_69_bar__minus_02_dot_53 loc_bar__minus_1_bar__minus_7_bar_2_bar_0)
        (receptacleAtLocation Shelf_bar__plus_01_dot_75_bar__plus_00_dot_17_bar__minus_02_dot_56 loc_bar_7_bar__minus_8_bar_2_bar_60)
        (receptacleAtLocation Shelf_bar__plus_01_dot_75_bar__plus_00_dot_55_bar__minus_02_dot_56 loc_bar_7_bar__minus_8_bar_2_bar_60)
        (receptacleAtLocation Shelf_bar__plus_01_dot_75_bar__plus_00_dot_88_bar__minus_02_dot_56 loc_bar_7_bar__minus_8_bar_2_bar_60)
        (receptacleAtLocation StoveBurner_bar__minus_00_dot_04_bar__plus_00_dot_92_bar__minus_02_dot_37 loc_bar_0_bar__minus_7_bar_2_bar_45)
        (receptacleAtLocation StoveBurner_bar__minus_00_dot_04_bar__plus_00_dot_92_bar__minus_02_dot_58 loc_bar_0_bar__minus_7_bar_2_bar_45)
        (receptacleAtLocation StoveBurner_bar__minus_00_dot_47_bar__plus_00_dot_92_bar__minus_02_dot_37 loc_bar__minus_2_bar__minus_7_bar_2_bar_45)
        (receptacleAtLocation StoveBurner_bar__minus_00_dot_47_bar__plus_00_dot_92_bar__minus_02_dot_58 loc_bar__minus_2_bar__minus_7_bar_2_bar_45)
        (receptacleAtLocation Toaster_bar__minus_01_dot_84_bar__plus_00_dot_90_bar__plus_00_dot_13 loc_bar__minus_5_bar_1_bar_3_bar_45)
        (objectAtLocation Egg_bar__minus_01_dot_44_bar__plus_00_dot_94_bar__minus_02_dot_41 loc_bar__minus_5_bar__minus_7_bar_2_bar_45)
        (objectAtLocation Egg_bar__plus_02_dot_06_bar__plus_00_dot_60_bar__minus_02_dot_56 loc_bar_7_bar__minus_8_bar_2_bar_60)
        )


        (:goal
            (and
                (exists (?o - object ?r - receptacle)
                    (and 
                        (heatable ?o)
                        (objectType ?o EggType) 
                        (receptacleType ?r CounterTopType)
                        (isHot ?o)
                        (inReceptacle ?o ?r) 
                    )
                )
            )
        )
    )
    