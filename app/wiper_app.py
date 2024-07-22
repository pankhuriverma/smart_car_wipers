# Copyright (c) 2022-2023 Robert Bosch GmbH and Microsoft Corporation
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""A sample Velocitas vehicle app for adjusting seat position."""

import json
import logging

from vehicle import Vehicle  # type: ignore
from velocitas_sdk.util.log import (  # type: ignore
    get_opentelemetry_log_factory,
    get_opentelemetry_log_format,
)
from velocitas_sdk.vdb.reply import DataPointReply
from velocitas_sdk.vehicle_app import VehicleApp, subscribe_topic

from wiper_model import SmartWiperModel

logging.setLogRecordFactory(get_opentelemetry_log_factory())
logging.basicConfig(format=get_opentelemetry_log_format())
logging.getLogger().setLevel("DEBUG")
logger = logging.getLogger(__name__)


class SmartWiperApp(VehicleApp):

    def __init__(self, vehicle_client: Vehicle):
        super().__init__()
        self.Vehicle = vehicle_client
        self.intensity = 0

    async def on_start(self):
        """Run when the vehicle app starts"""
        self.intensity = SmartWiperModel.predict()
        await self.Vehicle.Body.Raindetection.Intensity.set(self.intensity)
        await self.Vehicle.Body.Raindetection.Intensity.subscribe(
            self.on_rain_intensity_change
        )

    async def on_rain_intensity_change(self, data: DataPointReply):
        intensity = data.get(self.Vehicle.Body.Raindetection.Intensity).value
        if intensity >= 50:
            await self.Vehicle.Body.Windshield.Front.Wiping.Mode.set('Wipe')
            await self.Vehicle.Body.Windshield.Front.Wiping.System.ActualPosition.set(0)
            await self.Vehicle.Body.Windshield.Front.Wiping.System.TargetPosition.set(90)

        else:

            await self.Vehicle.Body.Windshield.Front.Wiping.Mode.set('Wipe')
            await self.Vehicle.Body.Windshield.Front.Wiping.System.ActualPosition.set(15)
            await self.Vehicle.Body.Windshield.Front.Wiping.System.TargetPosition.set(90)

        '''response_topic = "wiperSystem"
        await self.publish_event(
            response_topic,
            json.dumps(
                {"Mode": {data.get(self.Vehicle.Body.Windshield.Front.Wiping.Mode).value},
                "ActualPosition": {data.get(self.Vehicle.Body.Windshield.Front.Wiping.System.ActualPosition).value},
                "TargetPosition": {data.get(self.Vehicle.Body.Windshield.Front.Wiping.System.TargetPosition).value},}
            )'''


